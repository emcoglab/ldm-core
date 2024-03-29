"""
===========================
Testing against word association data.
===========================

Dr. Cai Wingfield
---------------------------
Embodied Cognition Lab
Department of Psychology
University of Lancaster
c.wingfield@lancaster.ac.uk
caiwingfield.net
---------------------------
2017
---------------------------
"""

from __future__ import annotations

import logging
import re
import csv
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
from os import path

import numpy
import statsmodels.formula.api as sm
from pandas import DataFrame

from .test import Test, Tester
from .results import EvaluationResults
from ..model.base import LinguisticDistributionalModel, VectorModel
from ..model.ngram import NgramModel
from ..utils.exceptions import WordNotFoundError
from ..utils.maths import DistanceType, CorrelationType
from ..preferences.preferences import Preferences

logger = logging.getLogger(__name__)


class AssociationResults(EvaluationResults):
    def __init__(self):
        super().__init__(
            results_column_names=["Correlation type", "Correlation", "Model BIC", "Baseline BIC", "B10 approx", "Log10 B10 approx"],
            save_dir=Preferences.association_results_dir
        )


class WordAssociationTest(Test, metaclass=ABCMeta):
    class TestColumn:
        word_1 = "Word 1"
        word_2 = "Word 2"
        association_strength = "Association strength"

    test_columns = [
        TestColumn.word_1,
        TestColumn.word_2,
        TestColumn.association_strength,
    ]

    def __init__(self, name: str):
        super().__init__(name)
        self.associations: List[WordAssociationTest.WordAssociation] = self._load()

    def associations_to_dataframe(self) -> DataFrame:
        return DataFrame.from_records(
            data=[
                {
                    WordAssociationTest.TestColumn.word_1: association.word_1,
                    WordAssociationTest.TestColumn.word_2: association.word_2,
                    WordAssociationTest.TestColumn.association_strength: association.association_strength,
                }
                for association in self.associations
            ],
            columns=WordAssociationTest.test_columns)

    @abstractmethod
    def _load(self) -> List[WordAssociationTest.WordAssociation]:
        raise NotImplementedError()

    @dataclass
    class WordAssociation(object):
        """A judgement of the association between two words."""
        word_1: str
        word_2: str
        association_strength: float

        def __str__(self):
            return f"{self.word_1}, {self.word_2}: {self.association_strength}"


# Static class
class AssociationTester(Tester):
    """
    Administers a word-association test against a model.
    """
    def __init__(self, test: WordAssociationTest, save_progress: bool = True, force_refresh: bool = False):
        self.test: WordAssociationTest = test
        super().__init__(save_progress, force_refresh)

    def _fresh_data(self) -> DataFrame:
        return self.test.associations_to_dataframe()

    @property
    def _save_path(self) -> str:
        return path.join(Preferences.association_results_dir, f"{self.test.name} data.csv")

    def has_tested_model(self,
                         model: LinguisticDistributionalModel,
                         distance_type: Optional[DistanceType] = None) -> bool:
        return self.column_name_for_model(model, distance_type) in self._data.columns.values

    def column_name_for_model(self,
                              model: LinguisticDistributionalModel,
                              distance_type: Optional[DistanceType]) -> str:
        self._validate_model_params(model, distance_type)
        if distance_type is None:
            return f"{model.name}"
        else:
            return f"{model.name}: {distance_type.name}"

    def administer_test(
            self,
            model: LinguisticDistributionalModel,
            distance_type: Optional[DistanceType] = None):
        """
        Administers a battery of tests against a model

        :param model: Must be trained.
        :param distance_type:
        """

        if distance_type is not None:
            logger.info(f"Administering {self.test.name} test with {model.name} and {distance_type.name}")
        else:
            logger.info(f"Administering {self.test.name} test with {model.name}")

        # validate args
        self._validate_model_params(model, distance_type)
        assert model.is_trained

        model_distance_col_name = self.column_name_for_model(model, distance_type)

        # Treat missing words as missing data
        def association_or_nan(word_pair):
            w1, w2 = word_pair
            try:
                if isinstance(model, NgramModel):
                    return model.association_between(w1, w2)
                elif isinstance(model, VectorModel):
                    return model.distance_between(w1, w2, distance_type)
                else:
                    raise NotImplementedError()
            except WordNotFoundError:
                return numpy.nan

        self._data[model_distance_col_name] = self._data[
            [WordAssociationTest.TestColumn.word_1, WordAssociationTest.TestColumn.word_2]
        ].apply(association_or_nan, axis=1)

        if self._save_progress:
            self._save_data()

    def results_for_model(self,
                          correlation_type: CorrelationType,
                          model: LinguisticDistributionalModel,
                          distance_type: Optional[DistanceType] = None) -> dict:
        """
        Save results based on the current self._data.
        Returns a dict ready to add to a AssociationResults.
        """
        assert self.has_tested_model(model, distance_type)

        local_data: DataFrame = self._data.copy()
        # Remove rows with missing results, as they wouldn't be missing in the baseline case.
        local_data.dropna(how="any")
        # Rename to make regression formulae easier
        local_data.rename(columns={
            self.column_name_for_model(model, distance_type): "model",
            WordAssociationTest.TestColumn.association_strength: "human",
        }, inplace=True)

        # Apply correlation
        if correlation_type == CorrelationType.Pearson:
            correlation = local_data["human"].corr(local_data["model"], method="pearson")
        elif correlation_type == CorrelationType.Spearman:
            correlation = local_data["human"].corr(local_data["model"], method="spearman")
        else:
            raise NotImplementedError()

        # Estimate Bayes factor from regression, as advised in
        # Jarosz & Wiley (2014) "What Are the Odds? A Practical Guide to Computing and Reporting Bayes Factors".
        # Journal of Problem Solving 7. doi:10.7771/1932-6246.1167. p. 5.

        # For spearman, we rank the data before regressing
        if correlation_type == CorrelationType.Spearman:
            local_data["human"] = local_data["human"].rank()
            local_data["model"] = local_data["model"].rank()

        # Compare variance explained (correlation squared) with two predictors versus one predictor (intercept)
        # To make data comparable, we need to drop rows with no model prediction
        local_data.dropna(subset=["model"], inplace=True)
        model_bic    = sm.ols(formula="human ~ model", data=local_data).fit().bic
        baseline_bic = sm.ols(formula="human ~ 1",     data=local_data).fit().bic
        b10_approx   = numpy.exp((baseline_bic - model_bic) / 2)
        # In case b10 goes to inf
        log10_b10_approx = ((baseline_bic - model_bic) / 2) * numpy.log10(numpy.exp(1))

        return {
            "Correlation type": correlation_type.name,
            "Correlation":      correlation,
            "Model BIC":        model_bic,
            "Baseline BIC":     baseline_bic,
            "B10 approx":       b10_approx,
            "Log10 B10 approx": log10_b10_approx
        }

    @staticmethod
    def _validate_model_params(model: LinguisticDistributionalModel, distance_type: Optional[DistanceType]):
        if isinstance(model, NgramModel):
            assert distance_type is None
        if isinstance(model, VectorModel):
            assert distance_type is not None


class SimlexSimilarity(WordAssociationTest):
    """Simlex-999 judgements."""

    def __init__(self):
        super().__init__("Simlex-999")

    def _load(self) -> List[WordAssociationTest.WordAssociation]:
        entry_re = re.compile(r"^"
                              r"(?P<word_1>[a-z]+)"  # The first concept in the pair.
                              r"\s+"
                              r"(?P<word_2>[a-z]+)"  # The second concept in the pair.
                              r"\s+"
                              r"(?P<pos_tag>[A-Z])"  # The majority part-of-speech of the concept words, from BNC.
                              r"\s+"
                              r"(?P<simlex_999>[0-9.]+)"  # The SimLex999 similarity rating.  In range [0, 10].
                              r"\s+"
                              r"(?P<conc_w1>[0-9.]+)"  # Concreteness of word 1, from SFFAN.  In range [1, 7].
                              r"\s+"
                              r"(?P<conc_w2>[0-9.]+)"  # Concreteness of word 2, from SFFAN.  In range [1, 7].
                              r"\s+"
                              r"(?P<conc_q>[0-9])"  # The quartile the pair occupies.
                              r"\s+"
                              r"(?P<assoc_usf>[0-9.]+)"  # Strength of free association from word 1 to 2, from SFFAN.
                              r"\s+"
                              r"(?P<sim_assoc_333>[01])"  # Whether pair is among 333 most associated in the dataset. 
                              r"\s+"
                              r"(?P<sd_simlex>[0-9.]+)"  # The standard deviation of annotator scores.
                              r"\s*$")

        with open(Preferences.simlex_path, mode="r", encoding="utf-8") as simlex_file:
            # Skip header line
            simlex_file.readline()

            associations = []
            for line in simlex_file:
                entry_match = re.match(entry_re, line)
                if entry_match:
                    associations.append(WordAssociationTest.WordAssociation(
                        entry_match.group("word_1"),
                        entry_match.group("word_2"),
                        float(entry_match.group("simlex_999"))))

        return associations


# TODO: rename to "MenRelatedness"
class MenSimilarity(WordAssociationTest):
    """
    MEN relatedness judgements.
    From: Bruni, E., Tran, NK., Baroni, M. "Multimodal Distributional Semantics". J. AI Research. 49:1--47 (2014).
    """
    def __init__(self):
        super().__init__("MEN")

    def _load(self) -> List[WordAssociationTest.WordAssociation]:
        entry_re = re.compile(r"^"
                              r"(?P<word_1>[a-z]+)"  # The first concept in the pair.
                              r"\s"
                              r"(?P<word_2>[a-z]+)"  # The second concept in the pair.
                              r"\s"
                              r"(?P<association>[0-9.]+)"  # Strength of association.
                              r"\s*$")

        with open(Preferences.men_path, mode="r", encoding="utf-8") as men_file:
            judgements = []
            for line in men_file:
                entry_match = re.match(entry_re, line)
                if entry_match:
                    judgements.append(WordAssociationTest.WordAssociation(
                        entry_match.group("word_1"),
                        entry_match.group("word_2"),
                        float(entry_match.group("association"))))

        return judgements


class WordsimSimilarity(WordAssociationTest):
    """WordSim-353 similarity judgements."""
    def __init__(self):
        super().__init__("WordSim-353 similarity")

    def _load(self) -> List[WordAssociationTest.WordAssociation]:
        entry_re = re.compile(r"^"
                              r"(?P<word_1>[a-z]+)"  # The first concept in the pair.
                              r"\s+"
                              r"(?P<word_2>[a-z]+)"  # The second concept in the pair.
                              r"\s+"
                              r"(?P<similarity>[0-9.]+)"  # The average similarity judgement.  In range [1, 10].
                              r"\s*$")

        with open(Preferences.wordsim_similarity_path, mode="r", encoding="utf-8") as wordsim_file:
            # Skip header line
            wordsim_file.readline()
            judgements = []
            for line in wordsim_file:
                entry_match = re.match(entry_re, line)
                if entry_match:
                    judgements.append(WordAssociationTest.WordAssociation(
                        entry_match.group("word_1"),
                        entry_match.group("word_2"),
                        float(entry_match.group("similarity"))))

        return judgements


class RelRelatedness(WordAssociationTest):
    """Rel-122 relatedness judgements."""
    def __init__(self):
        super().__init__("Rel-122 relatedness")

    def _load(self) -> List[WordAssociationTest.WordAssociation]:
        entry_re = re.compile(r"^"
                              r"(?P<relatedness>[0-9.]+)"  # The average relatedness judgement
                              r"\s+"
                              r"(?P<word_1>[a-z\- ]+)"  # The first concept in the pair
                              r"\s--\s"
                              r"(?P<word_2>[a-z\- ]+)"  # The second concept in the pair
                              r"\s*$")

        with open(Preferences.rel_path, mode="r", encoding="utf-8") as rel_file:
            judgements = []
            for line_i, line in enumerate(rel_file):
                entry_match = re.match(entry_re, line)
                if entry_match:
                    judgements.append(WordAssociationTest.WordAssociation(
                        entry_match.group("word_1"),
                        entry_match.group("word_2"),
                        float(entry_match.group("relatedness"))
                    ))
                else:
                    logger.warning(f"No match found on line {line_i}")
        return judgements


class SmallWorldOfWords(WordAssociationTest):
    """de Deyne et al.'s "Small World of Words" data."""

    class ResponsesType(Enum):
        # Just first responses
        R1   = 1
        # All three responses
        R123 = 2

    def __init__(self, responses_type: SmallWorldOfWords.ResponsesType):
        self._responses_type = responses_type
        super().__init__(f"Small World of Words {self._responses_type.name}")

    def _load(self):
        if self._responses_type == SmallWorldOfWords.ResponsesType.R1:
            file_path = Preferences.swow_r1_path
        elif self._responses_type == SmallWorldOfWords.ResponsesType.R123:
            file_path = Preferences.swow_r123_path
        else:
            raise NotImplementedError()
        with open(file_path, mode="r", encoding="utf-8") as swow_file:
            # Skip header line
            swow_file.readline()
            assocs = []
            for line in swow_file:
                parts = line.split("\t")
                assocs.append(WordAssociationTest.WordAssociation(
                    parts[0].lower().strip(),  # word 1
                    parts[1].lower().strip(),  # word 2
                    float(parts[4]),  # R123.Strength
                ))
        return assocs


class RubensteinGoodenough(WordAssociationTest):
    """Rubenstein & Goodenough (1965) synonymy test."""
    def __init__(self):
        super().__init__("Rubenstein & Goodenough")

    def _load(self) -> List[WordAssociationTest.WordAssociation]:
        with open(Preferences.rg65_path, mode="r", encoding="utf-8") as rg65_file:
            # Skip header line
            rg65_file.readline()
            assocs = []
            for line in rg65_file:
                parts = line.split(",")
                assocs.append(WordAssociationTest.WordAssociation(
                    parts[0].lower().strip(),  # word 1
                    parts[1].lower().strip(),  # word 2
                    float(parts[2].strip()),  # judged synonymy
                ))
        return assocs


class MillerCharlesSimilarity(WordAssociationTest):
    """Miller & Charles (1991) semantic similarity test."""
    def __init__(self):
        super().__init__("Miller & Charles")

    def _load(self) -> List[WordAssociationTest.WordAssociation]:
        with open(Preferences.miller_charles_path, mode="r", encoding="utf-8") as rg65_file:
            # Skip header line
            rg65_file.readline()
            assocs = []
            for line in rg65_file:
                parts = line.split(",")
                assocs.append(WordAssociationTest.WordAssociation(
                    parts[0].lower().strip(),  # word 1
                    parts[1].lower().strip(),  # word 2
                    float(parts[2].strip()),  # judged synonymy
                ))
        return assocs


class WordsimRelatedness(WordAssociationTest):
    """WordSim-353 relatedness judgements."""
    def __init__(self):
        super().__init__("WordSim-353 relatedness")

    def _load(self) -> List[WordAssociationTest.WordAssociation]:
        entry_re = re.compile(r"^"
                              r"(?P<word_1>[a-z]+)"  # The first concept in the pair.
                              r"\s+"
                              r"(?P<word_2>[a-z]+)"  # The second concept in the pair.
                              r"\s+"
                              r"(?P<relatedness>[0-9.]+)"  # The average relatedness judgement.  In range [1, 10].
                              r"\s*$")

        with open(Preferences.wordsim_relatedness_path, mode="r", encoding="utf-8") as wordsim_file:
            # Skip header line
            wordsim_file.readline()
            judgements = []
            for line in wordsim_file:
                entry_match = re.match(entry_re, line)
                if entry_match:
                    judgements.append(WordAssociationTest.WordAssociation(
                        entry_match.group("word_1"),
                        entry_match.group("word_2"),
                        float(entry_match.group("relatedness"))))

        return judgements


class WordsimAll(WordAssociationTest):
    """WordSim-353 all judgements."""
    def __init__(self):
        super().__init__("WordSim-353 all")

    def _load(self) -> List[WordAssociationTest.WordAssociation]:
        entry_re = re.compile(r"^"
                              r"(?P<word_1>[a-z]+)"  # The first concept in the pair.
                              r","
                              r"(?P<word_2>[a-z]+)"  # The second concept in the pair.
                              r","
                              r"(?P<relatedness>[0-9.]+)"  # The average relatedness judgement.  In range [1, 10].
                              r"\s*$")
        with open(Preferences.wordsim_all_path, mode="r", encoding="utf-8") as wordsim_file:
            # Skip header line
            wordsim_file.readline()
            judgements = []
            for line in wordsim_file:
                entry_match = re.match(entry_re, line.lower())
                if entry_match:
                    judgements.append(WordAssociationTest.WordAssociation(
                        entry_match.group("word_1"),
                        entry_match.group("word_2"),
                        float(entry_match.group("relatedness"))))
        return judgements


class ColourEmotionAssociation(WordAssociationTest):
    """
    Sutton & Altarriba (2016) colour–emotion association norms.
    """
    def __init__(self):
        super().__init__("Colour associations")

    def _load(self) -> List[WordAssociationTest.WordAssociation]:
        with open(Preferences.colour_association_path, mode="r", encoding="utf-8") as colour_assoc_file:
            # Skip header line
            colour_assoc_file.readline()
            assocs = []
            for line in colour_assoc_file:
                parts = line.split(",")
                assocs.append(WordAssociationTest.WordAssociation(
                    # word
                    parts[1].lower(),
                    # colour
                    parts[2].lower(),
                    # percentage of respondents
                    float(parts[4])))
        return assocs


class ThematicRelatedness(WordAssociationTest):
    """Jouravlev & McRae (2015) thematic relatedness production norms."""

    def __init__(self, only_use_response=None):
        """
        :param only_use_response: If None (default), use order-weighted response frequency.
        """
        assert only_use_response in [None, 1, 2, 3]
        self._only_use_response = only_use_response
        super().__init__("Thematic relatedness"
                         if only_use_response is None
                         else f"Thematic relatedness (R{only_use_response} only)")

    def _load(self) -> List[WordAssociationTest.WordAssociation]:
        with open(Preferences.thematic_association_path, mode="r", encoding="utf-8") as thematic_assoc_file:

            csvreader = csv.reader(thematic_assoc_file, delimiter=",", quotechar='"')

            assocs = []
            for line_i, line in enumerate(csvreader):

                # Skip header line
                if line_i == 0:
                    continue

                # Stop when last is reached
                if not line:
                    break

                word                      = line[0].lower().strip()
                response                  = line[1].lower().strip()
                respondent_count_r1       = int(line[2]) if line[2] else 0
                respondent_count_r2       = int(line[3]) if line[3] else 0
                respondent_count_r3       = int(line[4]) if line[4] else 0
                respondent_count_total    = int(line[5])
                respondent_count_weighted = int(line[6])

                # Check things went right and verify formulae used for summaries
                assert respondent_count_total == respondent_count_r1 + respondent_count_r2 + respondent_count_r3
                assert respondent_count_weighted == (3*respondent_count_r1) + (2*respondent_count_r2) + (1*respondent_count_r3)

                # Some responses have alternatives listed in brackets
                if "(" in response:
                    # Take only part of response before the alternatives
                    response = response.split("(")[0].strip()

                if self._only_use_response is None:
                    similarity_value = respondent_count_weighted
                elif self._only_use_response == 1:
                    similarity_value = respondent_count_r1
                elif self._only_use_response == 2:
                    similarity_value = respondent_count_r2
                elif self._only_use_response == 3:
                    similarity_value = respondent_count_r3
                else:
                    raise ValueError()

                assocs.append(WordAssociationTest.WordAssociation(
                    word,
                    response,
                    similarity_value))

        return assocs
