"""
===========================
Testing against human priming data.
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
import os
import pickle
import re
from abc import ABCMeta, abstractmethod
from typing import List, Set, Optional

from numpy import nan, exp, log10
from pandas import DataFrame, read_csv, merge, ExcelFile, Series
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from ..model.base import VectorModel, LinguisticDistributionalModel
from ..model.ngram import NgramModel
from ..model.predict import PredictVectorModel
from ..utils.exceptions import WordNotFoundError
from ..utils.maths import DistanceType
from ..preferences.preferences import Preferences

logger = logging.getLogger(__name__)


class RegressionData(metaclass=ABCMeta):
    """
    Regression data.
    """
    def __init__(self,
                 name: str,
                 pickle_path: str,
                 results_dir: str,
                 save_progress: bool = True,
                 force_reload:  bool = False):

        self.name: str = name

        self._pickle_path: str = pickle_path
        self._results_dir: str = results_dir

        self._save_progress: bool = save_progress

        # self._all_data backs self.dataframe
        # Load data if possible
        if self._could_load and not force_reload:
            logger.info(f"Loading previously saved {name} data")
            self._all_data = self._load()
        elif self._could_load_csv and not force_reload:
            logger.warning(f"Could not find previously saved data, attempting to rebuild from csv")
            self._all_data = self._load_from_csv()
        else:
            logger.info(f"Loading {name} data from source xls file")
            self._all_data = self._load_from_source()

        if self._save_progress:
            self.save()

    @property
    def dataframe(self) -> DataFrame:
        return self._all_data

    def _load(self) -> DataFrame:
        """
        Load previously saved data.
        """
        with open(self._pickle_path, mode="rb") as pickle_file:
            return pickle.load(pickle_file)

    def _save_pickle(self):
        """
        Save and overwrite data in pickle format.
        """
        assert self._all_data is not None
        with open(self._pickle_path, mode="wb") as pickle_file:
            pickle.dump(self._all_data, pickle_file)

    @property
    def _csv_path(self):
        """
        The filename of the exported CSV.
        """
        return os.path.join(self._results_dir, "model_predictors.csv")

    def export_csv(self, path: str = None):
        """
        Export the current dataframe as a csv.
        """
        assert self._all_data is not None

        if path is None:
            path = self._csv_path

        with open(path, mode="w", encoding="utf-8") as results_file:
            self.dataframe.to_csv(results_file, index=False)

    def _load_from_csv(self) -> DataFrame:
        """
        Load previously saved data from a CSV.
        """
        df = read_csv(self._csv_path, header=0, index_col=None, dtype={"Word": str})
        return df

    @property
    def _could_load(self) -> bool:
        """
        Whether data has been previously saved.
        """
        return os.path.isfile(self._pickle_path)

    @property
    def _could_load_csv(self) -> bool:
        """
        Whether the data has previously been exported as a csv.
        """
        return os.path.isfile(self._csv_path)

    @abstractmethod
    def _load_from_source(self) -> DataFrame:
        """
        Load data from the source data file, dealing with errors in source material.
        """
        raise NotImplementedError()

    def save(self):
        """
        Save and overwrite data.
        """
        if not self._save_progress:
            logger.warning("Tried to save progress with save_progress set to False. Not saving.")
            return
        self._save_pickle()
        self.export_csv()

    @property
    @abstractmethod
    def vocabulary(self) -> Set[str]:
        """
        The set of words used.
        """
        raise NotImplementedError()

    def missing_words(self, model: VectorModel) -> List[str]:
        """
        The list of SPP words which aren't present in a model.
        :type model: VectorModel
        :param model: Must be trained.
        :return: List of missing words.
        """
        assert model.is_trained

        missing_word_list = []
        for word in self.vocabulary:
            if not model.contains_word(word):
                missing_word_list.append(word)

        return sorted([w for w in self.vocabulary if not model.contains_word(w)])

    def predictor_exists_with_name(self, predictor_name: str) -> bool:
        """
        Whether the named predictor is already added.
        """
        return predictor_name in self.dataframe.columns.values

    def add_word_keyed_predictor(self, predictor: DataFrame, key_name: str, predictor_name: str):
        """
        Adds a word-keyed predictor column.
        :param predictor:
         Should have a column named `key_name`, used to left-join with the main dataframe, and a column named
         `predictor_name`, containing the actual values.
        :param predictor_name:
        :param key_name:
        :return:
        """

        # Skip the predictor if at already exists
        if self.predictor_exists_with_name(predictor_name):
            logger.info(f"Predictor '{predictor_name} already exists")
            return

        self._all_data = merge(self.dataframe, predictor, on=key_name, how="left")

        # Save in current state
        if self._save_progress:
            self.save()

    def add_word_pair_keyed_predictor(self, predictor: DataFrame, merge_on):
        """
        Adds a predictor column keyed from a prime-target pair.
        """

        self._all_data = merge(self.dataframe, predictor, on=merge_on, how="left")

        # Save in current state
        if self._save_progress:
            self.save()


class SppData(RegressionData):
    """
    Semantic Priming Project data.
    """
    class Columns:
        prime_word    = "PrimeWord"
        target_word   = "TargetWord"
        matched_prime = "MatchedPrime"
        prime_type    = "PrimeType"
        # ldt
        ldt_200_z = "LDT_200ms_Z"
        ldt_200_ac = "LDT_200ms_Acc"
        ldt_1200_z = "LDT_1200ms_Z"
        ldt_1200_ac = "LDT_1200ms_Acc"
        # nt
        nt_200_z = "NT_200ms_Z"
        nt_200_ac = "NT_200ms_Acc"
        nt_1200_z = "NT_1200ms_Z"
        nt_1200_ac = "NT_1200ms_Acc"
        # ldt priming
        ldt_200_z_priming = "LDT_200ms_Z_Priming"
        ldt_200_ac_priming = "LDT_200ms_Acc_Priming"
        ldt_1200_z_priming = "LDT_1200ms_Z_Priming"
        ldt_1200_ac_priming = "LDT_1200ms_Acc_Priming"
        # nt priming
        nt_200_z_priming = "NT_200ms_Z_Priming"
        nt_200_ac_priming = "NT_200ms_Acc_Priming"
        nt_1200_z_priming = "NT_1200ms_Z_Priming"
        nt_1200_ac_priming = "NT_1200ms_Acc_Priming"
        # baseline
        prime_length = "PrimeLength"
        target_length = "TargetLength"
        elex_prime_log_wf = "elex_prime_LgSUBTLWF"
        elex_prime_old20 = "elex_prime_OLD"
        elex_prime_pld20 = "elex_prime_PLD"
        elex_prime_nsyll = "elex_prime_NSyll"
        elex_target_log_wf = "elex_target_LgSUBTLWF"
        elex_target_old20 = "elex_target_OLD"
        elex_target_pld20 = "elex_target_PLD"
        elex_target_nsyll = "elex_target_NSyll"
        prime_target_old = "PrimeTarget_OrthLD"
        # baseline priming
        prime_target_old_priming = "PrimeTarget_OrthLD_Priming"
        # pos
        prime_pos = "PrimePOS"
        target_pos = "TargetPOS"

    def __init__(self,
                 save_progress: bool = True,
                 force_reload:  bool = False):
        super().__init__(name="SPP",
                         pickle_path=Preferences.spp_path_pickle,
                         results_dir=Preferences.spp_results_dir,
                         save_progress=save_progress,
                         force_reload=force_reload)

    def export_csv_first_associate_only(self, path=None):
        """
        Export the current dataframe as a csv, but only rows for the first associate primes.
        :param path: Save to specified path, else use default in Preferences.
        """
        assert self._all_data is not None
        results_csv_path = path if path is not None else os.path.join(self._results_dir, "model_predictors_first_associate_only.csv")
        first_assoc_data = self._all_data.query(f'{SppData.Columns.prime_type} == "first_associate"')
        with open(results_csv_path, mode="w", encoding="utf-8") as results_file:
            first_assoc_data.to_csv(results_file)

    @classmethod
    def _load_from_source(cls) -> DataFrame:
        prime_target_data: DataFrame = read_csv(Preferences.spp_path_csv, header=0)

        # Convert all to strings (to avoid False becoming a bool 😭)
        prime_target_data[SppData.Columns.target_word] = prime_target_data[SppData.Columns.target_word].apply(str)
        prime_target_data[SppData.Columns.prime_word] = prime_target_data[SppData.Columns.prime_word].apply(str)
        prime_target_data[SppData.Columns.matched_prime] = prime_target_data[SppData.Columns.matched_prime].apply(str)

        # Convert all to lower case
        prime_target_data[SppData.Columns.target_word] = prime_target_data[SppData.Columns.target_word].str.lower()
        prime_target_data[SppData.Columns.prime_word] = prime_target_data[SppData.Columns.prime_word].str.lower()
        prime_target_data[SppData.Columns.matched_prime] = prime_target_data[SppData.Columns.matched_prime].str.lower()

        # For unrelated pairs, the Matched Prime column will now have the string "nan".
        # There are no legitimate cases of "nan" as a matched prime.
        # So we go through and remove this.
        prime_target_data[SppData.Columns.matched_prime].replace("nan", nan, inplace=True)

        return prime_target_data

    @property
    def vocabulary(self) -> Set[str]:
        vocab: Set[str] = set()

        vocab = vocab.union(set(self.dataframe[SppData.Columns.prime_word]))
        vocab = vocab.union(set(self.dataframe[SppData.Columns.target_word]))

        return vocab

    @property
    def word_pairs(self) -> List[List[str]]:
        """
        Word pairs used in the SPP data.
        """
        return self.dataframe.reset_index()[[SppData.Columns.prime_word, SppData.Columns.target_word]].values.tolist()

    @classmethod
    def predictor_name_for_model(cls,
                                 model: LinguisticDistributionalModel,
                                 distance_type: Optional[DistanceType],
                                 for_priming_effect: bool) -> str:

        if distance_type is None:
            unsafe_name = f"{model.name}"
        else:
            unsafe_name = f"{model.name}_{distance_type.name}"

        # Remove unsafe characters
        unsafe_name = re.sub(r"[(),=]", "", unsafe_name)

        # Convert hyphens and spaces to underscores
        safe_name = re.sub(r"[-\s]", "_", unsafe_name)

        if for_priming_effect:
            safe_name = safe_name + "_Priming"

        return safe_name

    def add_model_predictor(self,
                            model: LinguisticDistributionalModel,
                            distance_type: Optional[DistanceType],
                            for_priming_effect: bool,
                            memory_map: bool = False):
        """
        Adds a data column containing predictors from a semantic model.
        """

        predictor_name = self.predictor_name_for_model(model, distance_type, for_priming_effect)

        # Skip existing predictors
        if self.predictor_exists_with_name(predictor_name):
            logger.info(f"Model predictor '{predictor_name}' already added")

        else:
            logger.info(f"Adding '{predictor_name}' model predictor")

            # Since we're going to use the model, make sure it's trained
            model.train(memory_map=memory_map)

            if for_priming_effect:
                # Make sure the non-priming model predictor exists already, as we'll be referencing it
                assert self.predictor_exists_with_name(self.predictor_name_for_model(model, distance_type,
                                                                                     for_priming_effect=False))

            def model_association_or_none(word_pair):
                """
                Get the association distance between a pair of words, or None, if one of the words doesn't exist.
                """
                word_1, word_2 = word_pair
                try:
                    # Vector models compare words using distances
                    if isinstance(model, VectorModel):
                        # The above type check should ensure that the model has this method.
                        # I think this warning and the following one are due to model being captured.
                        return model.distance_between(word_1, word_2, distance_type)
                    # Ngram models compare words using associations
                    elif isinstance(model, NgramModel):
                        return model.association_between(word_1, word_2)
                    else:
                        raise TypeError()
                except WordNotFoundError as er:
                    logger.warning(er.message)
                    return None

            # If we're computing the priming predictor, we'll find the matched-unrelated word, and
            # subtract the model distance of that from the model distance for the matched target-prime
            # pair.
            #
            # We're assuming that the matched predictor has already been added, so we can safely join
            # on the matched prime pair here, since there'll be a PrimeWord-matched predictor there
            # already.
            key_column = SppData.Columns.matched_prime if for_priming_effect else SppData.Columns.prime_word

            # Add model distance column to data frame
            model_association = self.dataframe[
                [key_column, SppData.Columns.target_word]
            ].apply(model_association_or_none, axis=1)

            if for_priming_effect:
                # The priming predictor is the difference in model distance between the related and
                # matched-unrelated word pairs.
                self.dataframe[predictor_name] = (
                        # The model association between the MATCHED prime word and the target word
                        model_association
                        # The already-calculated model association between the prime word and the target word
                        - self.dataframe[self.predictor_name_for_model(model, distance_type, for_priming_effect=False)])
            else:
                self.dataframe[predictor_name] = model_association

            # Save in current state
            if self._save_progress:
                self.save()


class CalgaryData(RegressionData):
    """
    Calgary data.
    """
    class Columns:
        word = "Word"
        word_type = "WordType"
        accuracy = "ACC"
        # data
        zrt_mean = "zRTclean_mean"
        concrete_response_proportion = "Concrete_response_proportion"
        # baseline
        elex_length = "elex_Length"
        elex_log_wf = "elex_LgSUBTLWF"
        elex_old20 = "elex_OLD"
        elex_pld20 = "elex_PLD"
        elex_nsyll = "elex_NSyll"
        concrete_old = "concrete_OrthLD"
        abstract_old = "abstract_OrthLD"

    def __init__(self,
                 save_progress: bool = True,
                 force_reload:  bool = False):
        super().__init__(name="Calgary",
                         pickle_path=Preferences.calgary_path_pickle,
                         results_dir=Preferences.calgary_results_dir,
                         save_progress=save_progress,
                         force_reload=force_reload)
        self._add_response_columns()

    @classmethod
    def _load_from_source(cls) -> DataFrame:
        """
        Load data from excel file, dealing with errors in source material.
        """
        xls = ExcelFile(Preferences.calgary_path_xlsx)
        word_data = xls.parse("Sheet1")

        # Convert all to strings (to avoid False becoming a bool 😭)
        word_data[CalgaryData.Columns.word] = word_data[CalgaryData.Columns.word].apply(str)

        # Convert all to lower case
        word_data[CalgaryData.Columns.word] = word_data[CalgaryData.Columns.word].str.lower()

        return word_data

    def _add_response_columns(self):
        """
        Adds a columns containing the fraction of respondents who answered concrete or abstract.
        """

        def concreteness_proportion(r) -> float:
            # If the word is Brysbaert-concrete, the accuracy equals the fraction of responders who decided "concrete"
            if r[CalgaryData.Columns.word_type] == "Concrete":
                return r[CalgaryData.Columns.accuracy]
            # If the word is Brysbaert-abstract, the complement of the accuracy equals the fraction of responders who
            # decided "concrete"
            else:
                return 1 - r[CalgaryData.Columns.accuracy]

        def abstractness_proportion(r) -> float:
            return 1-concreteness_proportion(r)

        if self.predictor_exists_with_name("Concrete_response_proportion"):
            logger.info("Concrete_response_proportion column already exists")
        else:
            logger.info("Adding Concrete_response_proportion column")
            self.dataframe["Concrete_response_proportion"] = self.dataframe.apply(concreteness_proportion, axis=1)

        if self.predictor_exists_with_name("Abstract_response_proportion"):
            logger.info("Abstract_response_proportion column already exists")
        else:
            logger.info("Adding Abstract_response_proportion column")
            self.dataframe["Abstract_response_proportion"] = self.dataframe.apply(abstractness_proportion, axis=1)

        if self._save_progress:
            self.save()

    @property
    def vocabulary(self) -> Set[str]:
        """The set of words used in the SPP data."""
        return set(self.dataframe[CalgaryData.Columns.word])

    @classmethod
    def predictor_name_for_model_fixed_reference(cls,
                                                 model: LinguisticDistributionalModel,
                                                 distance_type: Optional[DistanceType],
                                                 reference_word: str) -> str:
        if distance_type is None:
            unsafe_name = f"{model.name}_{reference_word}_distance"
        else:
            unsafe_name = f"{model.name}_{distance_type.name}_{reference_word}_distance"

        # Remove unsafe characters
        unsafe_name = re.sub(r"[(),=]", "", unsafe_name)

        # Convert hyphens and spaces to underscores
        safe_name = re.sub(r"[-\s]", "_", unsafe_name)

        return safe_name

    @property
    def reference_words(self) -> List[str]:
        return ["concrete", "abstract"]

    def add_model_predictor_fixed_reference(self,
                                            model: LinguisticDistributionalModel,
                                            distance_type: Optional[DistanceType],
                                            reference_word: str,
                                            memory_map: bool = False):
        """
        Adds a data column containing predictors from a semantic model.
        """

        predictor_name = f"{self.predictor_name_for_model_fixed_reference(model, distance_type, reference_word)}"

        # Skip existing predictors
        if self.predictor_exists_with_name(predictor_name):
            logger.info(f"Model predictor '{predictor_name}' already added")
            return

        else:
            logger.info(f"Adding '{predictor_name}' model predictor")

            # Since we're going to use the model, make sure it's trained
            model.train(memory_map=memory_map)

            def fixed_model_association_or_none(word):
                """
                Get the model distance between a pair of words, or None, if one of the words doesn't exist.
                """
                try:
                    # Vector models compare words using distances
                    if isinstance(model, VectorModel):
                        # The above type check should ensure that the model has this method.
                        # I think this warning and the following one are due to model being captured.
                        return model.distance_between(word, reference_word, distance_type)
                        # Ngram models compare words using associations
                    elif isinstance(model, NgramModel):
                        return model.association_between(word, reference_word)
                    else:
                        raise TypeError()
                except WordNotFoundError as er:
                    logger.warning(er.message)
                    return None

            # Add model distance column to data frame
            self.dataframe[predictor_name] = (
                self.dataframe[CalgaryData.Columns.word]
                .apply(fixed_model_association_or_none))

            # Save in current state
            if self._save_progress:
                self.save()


class RegressionResult(object):
    """
    The result of a priming regression.
    """
    def __init__(self,
                 dv_name: str,
                 model: LinguisticDistributionalModel,
                 distance_type: Optional[DistanceType],
                 baseline_r2: float,
                 baseline_bic: float,
                 model_r2: float,
                 model_bic: float,
                 model_t: float,
                 model_p: float,
                 model_beta: float,
                 df: int,
                 max_vif: float,
                 max_vif_predictor: str):

        # Dependent variable
        self.dv_name          = dv_name

        # Baseline R^2 from lexical factors
        self.baseline_r2      = baseline_r2

        # Model info
        self.model_type_name  = model.model_type.name
        self.embedding_size   = model.embedding_size if isinstance(model, PredictVectorModel) else None
        self.window_radius    = model.window_radius
        self.distance_type    = distance_type
        self.corpus_name      = model.corpus_meta.name

        # R^2 with the inclusion of the model predictors
        self.model_r2         = model_r2

        # Bayes information criteria and Bayes factors
        self.baseline_bic     = baseline_bic
        self.model_bic        = model_bic
        self.b10_approx       = exp((baseline_bic - model_bic) / 2)
        # Sometimes when BICs get big, B10 ends up at inf; but it's approx log10 should be finite
        # (and useful for ordering as log is monotonic increasing)
        self.b10_log_fallback = ((baseline_bic - model_bic) / 2) * log10(exp(1))

        # t, p, beta
        self.model_t          = model_t
        self.model_p          = model_p
        self.model_beta       = model_beta

        # Degrees of freedom
        self.df               = df

        # Variance inflation diagnostics
        self.max_vif             = max_vif
        self.max_vif_predictor   = max_vif_predictor

    @property
    def model_r2_increase(self) -> float:
        return self.model_r2 - self.baseline_r2

    @classmethod
    def headings(cls) -> List[str]:
        return [
            'Dependent variable',
            'Model type',
            'Embedding size',
            'Window radius',
            'Distance type',
            'Corpus',
            'Baseline R-squared',
            'Model R-squared',
            'R-squared increase',
            'Baseline BIC',
            'Model BIC',
            'B10 approx',
            'Log10 B10 approx',
            't',
            'p',
            'beta',
            'df',
            'max vif',
            'max vif predictor'
        ]

    @property
    def fields(self) -> List[str]:
        return [
            self.dv_name,
            self.model_type_name,
            str(self.embedding_size) if self.embedding_size is not None else "",
            str(self.window_radius),
            self.distance_type.name if self.distance_type is not None else "",
            self.corpus_name,
            str(self.baseline_r2),
            str(self.model_r2),
            str(self.model_r2_increase),
            str(self.baseline_bic),
            str(self.model_bic),
            str(self.b10_approx),
            str(self.b10_log_fallback),
            str(self.model_t),
            str(self.model_p),
            str(self.model_beta),
            str(self.df),
            str(self.max_vif),
            self.max_vif_predictor,
        ]


def variance_inflation_factors(exog: DataFrame):
    """
    Compute variance inflation factors from a design matrix.

    :param:
        exog : DataFrame, (nobs, k_vars)
            Design matrix with all explanatory variables, as for example used in regression.
    :return:
        max_vif : Series
    """
    exog = add_constant(exog)
    vifs = Series(
        [1.1 / (1.0 - OLS(exog[col].values,
                          exog.loc[:, exog.columns != col].values).fit().rsquared)
         for col in exog],
        index=exog.columns,
        name='VIF'
    )
    return vifs
