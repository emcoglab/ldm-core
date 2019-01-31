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

import logging
import re
import csv
from abc import ABCMeta, abstractmethod
from typing import List, Optional
from os import path

import numpy
import scipy.stats
import statsmodels.formula.api as sm
from pandas import DataFrame

from ..evaluation.results import EvaluationResults
from ..model.base import DistributionalSemanticModel, VectorSemanticModel
from ..model.ngram import NgramModel
from ..utils.exceptions import WordNotFoundError
from ..utils.maths import DistanceType, CorrelationType
from ...preferences.preferences import Preferences

logger = logging.getLogger(__name__)


class AssociationResults(EvaluationResults):
    def __init__(self):
        super().__init__(
            results_column_names=["Correlation type", "Correlation", "Model BIC", "Baseline BIC", "B10 approx", "Log10 B10 approx"],
            save_dir=Preferences.association_results_dir
        )


class WordAssociation(object):
    """
    A judgement of the similarity between two words.
    """

    def __init__(self, word_1: str, word_2: str, association_strength: float):
        self.word_1 = word_1
        self.word_2 = word_2
        self.association_strength = association_strength


class WordAssociationTest(metaclass=ABCMeta):
    """
    A test against word-association data.
    """

    def __init__(self):
        # Backs self.judgement_list
        self._association_list: List[WordAssociation] = None

    @property
    def association_list(self) -> List[WordAssociation]:
        """
        The list of associations.
        """
        # Lazy load
        if self._association_list is None:
            self._association_list = self._load()
        assert self._association_list is not None
        return self._association_list

    @property
    @abstractmethod
    def name(self) -> str:
        """
        The name of the test.
        """
        raise NotImplementedError()

    @abstractmethod
    def _load(self) -> List[WordAssociation]:
        """
        Load from source.
        """
        raise NotImplementedError()


# Static class
class AssociationTester(object):
    """
    Administers a word-association test against a model.
    """

    @staticmethod
    def administer_test(
            test: WordAssociationTest,
            model: DistributionalSemanticModel,
            distance_type: Optional[DistanceType]
            ) -> AssociationResults:
        """
        Administers a battery of tests against a model
        :param test:
        :param model: Must be trained.
        :param distance_type:
        """

        assert model.is_trained

        results = AssociationResults()

        human_judgements: List[WordAssociation] = []
        model_judgements: List[WordAssociation] = []
        for human_judgement in test.association_list:
            try:
                # Vector models compare words using distances
                if isinstance(model, VectorSemanticModel):
                    model_judgement = model.distance_between(
                        human_judgement.word_1,
                        human_judgement.word_2,
                        distance_type)
                # Ngram models compare words using associations
                elif isinstance(model, NgramModel):
                    model_judgement = model.association_between(
                        human_judgement.word_1,
                        human_judgement.word_2
                    )
                else:
                    raise TypeError()
            except WordNotFoundError as er:
                # If we can't find one of the words in the corpus, just ignore it.
                logger.warning(er.message)
                continue

            # If both words were found in the model, add them to the test list
            human_judgements.append(human_judgement)
            model_judgements.append(WordAssociation(
                human_judgement.word_1,
                human_judgement.word_2,
                model_judgement))

        # Save transcript
        if distance_type is None:
            transcript_csv_name = f"transcript test={test.name} model={model.name}.csv"
        else:
            transcript_csv_name = f"transcript test={test.name} model={model.name} distance={distance_type.name}.csv"

        transcript_csv_path = path.join(Preferences.association_results_dir, "transcripts", transcript_csv_name)
        DataFrame.from_dict({
            "Word 1"         : [j.word_1 for j in human_judgements],
            "Word 2"         : [j.word_2 for j in human_judgements],
            "Model distance" : [j.association_strength for j in model_judgements],
            "Data similarity": [j.association_strength for j in human_judgements]
        }).to_csv(transcript_csv_path, index=False)

        # Apply correlations
        for correlation_type in CorrelationType:
            if correlation_type is CorrelationType.Pearson:
                correlation = numpy.corrcoef(
                    [j.association_strength for j in human_judgements],
                    [j.association_strength for j in model_judgements])[0][1]

                # Estimate Bayes factor from regression, as advised in
                # Jarosz & Wiley (2014) "What Are the Odds? A Practical Guide to Computing and Reporting Bayes Factors".
                # Journal of Problem Solving 7. doi:10.7771/1932-6246.1167. p. 5.
                data: DataFrame = DataFrame.from_dict({
                    "human": [j.association_strength for j in human_judgements],
                    "model": [j.association_strength for j in model_judgements]
                })
                # Remove rows with missing results, as they wouldn't be missing in the baseline case.
                data = data.dropna(how="any")
                # Compare variance explained (correlation squared) with two predictors versus one predictor (intercept)
                model_bic    = sm.ols(formula="human ~ model", data=data).fit().bic
                baseline_bic = sm.ols(formula="human ~ 1",     data=data).fit().bic
                b10_approx   = numpy.exp((baseline_bic - model_bic) / 2)
                # In case b10 goes to inf
                log10_b10_approx = ((baseline_bic - model_bic) / 2) * numpy.log10(numpy.exp(1))
            elif correlation_type is CorrelationType.Spearman:
                # PyCharm erroneously detects input types for scipy.stats.spearmanr as int rather than ndarray
                # noinspection PyTypeChecker,PyUnresolvedReferences
                correlation = scipy.stats.spearmanr(
                    [j.association_strength for j in human_judgements],
                    [j.association_strength for j in model_judgements]).correlation

                # Estimate Bayes factors using same approach as for Pearson, but with ranked data
                # Since Spearman is just Pearson on ranks.
                data = DataFrame.from_dict({
                    "human": scipy.stats.rankdata([j.association_strength for j in human_judgements]),
                    "model": scipy.stats.rankdata([j.association_strength for j in model_judgements])
                })
                # Remove rows with missing results, as they wouldn't be missing in the baseline case.
                data = data.dropna(how="any")
                model_bic    = sm.ols(formula="human ~ model", data=data).fit().bic
                baseline_bic = sm.ols(formula="human ~ 1",     data=data).fit().bic
                b10_approx = numpy.exp((baseline_bic - model_bic) / 2)
                # In case b10 goes to inf
                log10_b10_approx = ((baseline_bic - model_bic) / 2) * numpy.log10(numpy.exp(1))
            else:
                raise ValueError(correlation_type)

            results.add_result(test.name, model, distance_type, {
                "Correlation type"  : correlation_type.name,
                "Correlation"       : correlation,
                "Model BIC"         : model_bic,
                "Baseline BIC"      : baseline_bic,
                "B10 approx"        : b10_approx,
                "Log10 B10 approx"  : log10_b10_approx
            })

        return results


class SimlexSimilarity(WordAssociationTest):
    """
    Simlex-999 judgements.
    """

    @property
    def name(self) -> str:
        return "Simlex-999"

    def _load(self) -> List[WordAssociation]:

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
                    associations.append(WordAssociation(
                        entry_match.group("word_1"),
                        entry_match.group("word_2"),
                        float(entry_match.group("simlex_999"))))

        return associations


class MenSimilarity(WordAssociationTest):
    """
    MEN similarity judgements.
    From: Bruni, E., Tran, NK., Baroni, M. "Multimodal Distributional Semantics". J. AI Research. 49:1--47 (2014).
    """

    @property
    def name(self) -> str:
        return "MEN"

    def _load(self) -> List[WordAssociation]:

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
                    judgements.append(WordAssociation(
                        entry_match.group("word_1"),
                        entry_match.group("word_2"),
                        float(entry_match.group("association"))))

        return judgements


class WordsimSimilarity(WordAssociationTest):
    """
    WordSim-353 similarity judgements.
    """

    @property
    def name(self) -> str:
        return "WordSim-353 similarity"

    def _load(self):
        entry_re = re.compile(r"^"
                              r"(?P<word_1>[a-z]+)"  # The first concept in the pair.
                              r"\s+"
                              r"(?P<word_2>[a-z]+)"  # The second concept in the pair.
                              r"\s+"
                              # TODO: is it really in this range?
                              r"(?P<similarity>[0-9.]+)"  # The average similarity judgement.  In range [1, 10].
                              r"\s*$")

        with open(Preferences.wordsim_similarity_path, mode="r", encoding="utf-8") as wordsim_file:
            # Skip header line
            wordsim_file.readline()
            judgements = []
            for line in wordsim_file:
                entry_match = re.match(entry_re, line)
                if entry_match:
                    judgements.append(WordAssociation(
                        entry_match.group("word_1"),
                        entry_match.group("word_2"),
                        float(entry_match.group("similarity"))))

        return judgements


class WordsimRelatedness(WordAssociationTest):
    """
    WordSim-353 relatedness judgements.
    """

    @property
    def name(self) -> str:
        return "WordSim-353 relatedness"

    def _load(self):
        entry_re = re.compile(r"^"
                              r"(?P<word_1>[a-z]+)"  # The first concept in the pair.
                              r"\s+"
                              r"(?P<word_2>[a-z]+)"  # The second concept in the pair.
                              r"\s+"
                              # TODO: is it really in this range? Does it matter?
                              r"(?P<relatedness>[0-9.]+)"  # The average relatedness judgement.  In range [1, 10].
                              r"\s*$")

        with open(Preferences.wordsim_relatedness_path, mode="r", encoding="utf-8") as wordsim_file:
            # Skip header line
            wordsim_file.readline()
            judgements = []
            for line in wordsim_file:
                entry_match = re.match(entry_re, line)
                if entry_match:
                    judgements.append(WordAssociation(
                        entry_match.group("word_1"),
                        entry_match.group("word_2"),
                        float(entry_match.group("relatedness"))))

        return judgements


class ColourEmotionAssociation(WordAssociationTest):
    """
    Sutton & Altarriba (2016) colour–emotion association norms.
    """

    @property
    def name(self) -> str:
        return "Colour associations"

    def _load(self) -> List[WordAssociation]:
        with open(Preferences.colour_association_path, mode="r", encoding="utf-8") as colour_assoc_file:
            # Skip header line
            colour_assoc_file.readline()
            assocs = []
            for line in colour_assoc_file:
                parts = line.split(",")
                assocs.append(WordAssociation(
                    # word
                    parts[1].lower(),
                    # colour
                    parts[2].lower(),
                    # percentage of respondents
                    float(parts[4])))

        return assocs


class ThematicRelatedness(WordAssociationTest):
    """
    Jouravlev & McRae (2015) thematic relatedness production norms.
    """

    def __init__(self, only_use_response=None):
        """
        :param only_use_response: If None, use order-weighted response frequency.
        """
        super().__init__()
        assert only_use_response is None or only_use_response in [1, 2, 3]
        self._only_use_response = only_use_response

    @property
    def name(self) -> str:
        if self._only_use_response is None:
            return "Thematic relatedness"
        else:
            return f"Thematic relatedness (R{self._only_use_response} only)"

    def _load(self):
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

                assocs.append(WordAssociation(
                    word,
                    response,
                    similarity_value))

        return assocs
