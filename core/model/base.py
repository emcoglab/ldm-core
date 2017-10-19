"""
===========================
Base classes for language models.
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

from abc import ABCMeta, abstractmethod
from enum import Enum, auto

import numpy

from ..corpus.corpus import CorpusMetadata
from ..utils.maths import DistanceType, distance
from ...preferences.preferences import Preferences

logger = logging.getLogger(__name__)


class DistributionalSemanticModel(metaclass=ABCMeta):
    """
    A model of the language.
    """

    class MetaType(Enum):
        count = auto()
        predict = auto()

    class ModelType(Enum):
        """
        Representative of the type of a vector space model.
        """
        # Predict model
        cbow = auto()
        skip_gram = auto()

        # Count model
        ngram_unsummed = auto()
        ngram = auto()
        log_ngram = auto()
        ngram_probability = auto()
        token_probability = auto()
        context_probability = auto()
        conditional_probability = auto()
        probability_ratios = auto()
        ppmi = auto()

        @property
        def metatype(self):
            """
            The metatype of this type.
            :return:
            """
            if self is VectorSemanticModel.ModelType.cbow:
                return VectorSemanticModel.MetaType.predict
            elif self is VectorSemanticModel.ModelType.skip_gram:
                return VectorSemanticModel.MetaType.predict
            elif self is VectorSemanticModel.ModelType.ngram_unsummed:
                return VectorSemanticModel.MetaType.count
            elif self is VectorSemanticModel.ModelType.ngram:
                return VectorSemanticModel.MetaType.count
            elif self is VectorSemanticModel.ModelType.log_ngram:
                return VectorSemanticModel.MetaType.count
            elif self is VectorSemanticModel.ModelType.ngram_probability:
                return VectorSemanticModel.MetaType.count
            elif self is VectorSemanticModel.ModelType.token_probability:
                return VectorSemanticModel.MetaType.count
            elif self is VectorSemanticModel.ModelType.context_probability:
                return VectorSemanticModel.MetaType.count
            elif self is VectorSemanticModel.ModelType.conditional_probability:
                return VectorSemanticModel.MetaType.count
            elif self is VectorSemanticModel.ModelType.probability_ratios:
                return VectorSemanticModel.MetaType.count
            elif self is VectorSemanticModel.ModelType.pmi:
                return VectorSemanticModel.MetaType.count
            elif self is VectorSemanticModel.ModelType.ppmi:
                return VectorSemanticModel.MetaType.count
            else:
                raise ValueError()

        @property
        def slug(self):
            """
            A path-safe representation of the model type
            :return:
            """
            if self is DistributionalSemanticModel.ModelType.cbow:
                return "cbow"
            elif self is DistributionalSemanticModel.ModelType.skip_gram:
                return "skipgram"
            elif self is DistributionalSemanticModel.ModelType.ngram_unsummed:
                return "ngram_unsummed"
            elif self is DistributionalSemanticModel.ModelType.ngram:
                return "ngram"
            elif self is DistributionalSemanticModel.ModelType.log_ngram:
                return "log_ngram"
            elif self is DistributionalSemanticModel.ModelType.ngram_probability:
                return "ngram_probability"
            elif self is DistributionalSemanticModel.ModelType.token_probability:
                return "token_probability"
            elif self is DistributionalSemanticModel.ModelType.context_probability:
                return "context_probability"
            elif self is DistributionalSemanticModel.ModelType.conditional_probability:
                return "conditional_probability"
            elif self is DistributionalSemanticModel.ModelType.probability_ratios:
                return "probability_ratios"
            elif self is DistributionalSemanticModel.ModelType.ppmi:
                return "ppmi"
            else:
                raise ValueError()

        @property
        def name(self):
            """
            The name of the model type
            :return:
            """
            if self is VectorSemanticModel.ModelType.cbow:
                return "CBOW"
            elif self is VectorSemanticModel.ModelType.skip_gram:
                return "Skip-gram"
            elif self is VectorSemanticModel.ModelType.ngram_unsummed:
                return "n-gram (unsummed)"
            elif self is VectorSemanticModel.ModelType.ngram:
                return "n-gram (summed)"
            elif self is VectorSemanticModel.ModelType.log_ngram:
                return "log n-gram"
            elif self is VectorSemanticModel.ModelType.ngram_probability:
                return "n-gram probability"
            elif self is VectorSemanticModel.ModelType.token_probability:
                return "Token probability"
            elif self is VectorSemanticModel.ModelType.context_probability:
                return "Context probability"
            elif self is VectorSemanticModel.ModelType.conditional_probability:
                return "Conditional probability"
            elif self is VectorSemanticModel.ModelType.probability_ratios:
                return "Probability ratio"
            elif self is VectorSemanticModel.ModelType.ppmi:
                return "PPMI"
            else:
                raise ValueError()

        @classmethod
        def from_slug(cls, slug: str) -> 'DistributionalSemanticModel.ModelType':
            """
            Get the model type from the slug.
            """
            slug = slug.lower()
            if slug == "cbow":
                return cls.cbow
            elif slug == "skip-gram":
                return cls.skip_gram
            elif slug == "n-gram (unsummed)":
                return cls.ngram_unsummed
            elif slug == "n-gram (summed)":
                return cls.ngram
            elif slug == "log n-gram":
                return cls.log_ngram
            elif slug == "n-gram probability":
                return cls.ngram_probability
            elif slug == "token probability":
                return cls.token_probability
            elif slug == "context probability":
                return cls.context_probability
            elif slug == "conditional probability":
                return cls.conditional_probability
            elif slug == "probability ratio":
                return cls.probability_ratios
            elif slug == "ppmi":
                return cls.ppmi
            else:
                raise ValueError()

        @classmethod
        def predict_types(cls):
            """
            Lists the predict types
            :return:
            """
            return [t for t in VectorSemanticModel.ModelType if t.metatype is VectorSemanticModel.MetaType.predict]

        @classmethod
        def count_types(cls):
            """
            Lists the count types
            :return:
            """
            return [t for t in VectorSemanticModel.ModelType if t.metatype is VectorSemanticModel.MetaType.count]

    def __init__(self, model_type: ModelType, corpus_meta: CorpusMetadata):

        self.model_type = model_type
        self.corpus_meta = corpus_meta

    @property
    @abstractmethod
    def is_trained(self) -> bool:
        """
        True if the model the model data is present and ready to be queried.
        """
        raise NotImplementedError()

    @property
    def _root_dir(self) -> str:
        """
        The root directory for all models.
        """
        # We need to remember the root directory for all models, as well as the save directory for this model.
        # This allows us to instantiate and load other models from the correct root.
        return Preferences.model_dir

    @property
    def name(self) -> str:
        """
        The name of the model, containing all relevant information to disambiguate it from other models.
        """
        return f"{self.model_type.name} ({self.corpus_meta.name})"

    @property
    @abstractmethod
    def _model_filename(self) -> str:
        """
        The file name of the model, without file extension.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def _model_ext(self) -> str:
        """
        The file extension of the model file.
        """
        raise NotImplementedError()

    @property
    def _model_filename_with_ext(self) -> str:
        """
        The filename of the model, with extension
        """
        return self._model_filename + self._model_ext

    @property
    def save_dir(self) -> str:
        return os.path.join(self._root_dir, self.model_type.slug)

    @property
    def could_load(self) -> bool:
        """
        Whether or not a previously saved model exists on the drive.
        """
        return os.path.isfile(os.path.join(self.save_dir, self._model_filename_with_ext))

    def train(self, force_retrain: bool = False, memory_map: bool = False):
        """
        Trains the model from its corpus, and saves the resultant state to drive.
        Will load existing model instead if possible.
        :param force_retrain: Retrain the model, even if there is a pre-existing saved state. Default False.
        :param memory_map: Whether to load the model memory-mapped when loading. Default False.
        """
        if self.is_trained and not force_retrain:
            logger.info(f"{self.name} is already trained")
        elif self.could_load and not force_retrain:
            logger.info(f"Loading {self.name} model from {self._model_filename_with_ext}")
            self._load(memory_map=memory_map)
        else:
            logger.info(f"Training {self.name}")
            self._retrain()
            logger.info(f"Saving {self.name} model to {self._model_filename_with_ext}")
            self._save()

    @abstractmethod
    def _retrain(self):
        """
        Retrains a model from scratch.
        """
        raise NotImplementedError()

    @abstractmethod
    def _load(self, memory_map: bool = False):
        """
        Loads a model.
        """
        raise NotImplementedError()

    @abstractmethod
    def _save(self):
        """
        Saves a model in its current state.
        """
        raise NotImplementedError()

    @abstractmethod
    def contains_word(self, word: str) -> bool:
        """
        Whether the model is trained on a corpus containing a specific word.
        """
        raise NotImplementedError()


class VectorSemanticModel(DistributionalSemanticModel, metaclass=ABCMeta):
    """
    A language model where each word is associated with a point in a vector space.
    """

    def __init__(self,
                 model_type: DistributionalSemanticModel.ModelType,
                 corpus_meta: CorpusMetadata,
                 window_radius: int):
        super().__init__(model_type, corpus_meta)
        self.window_radius = window_radius

        # When implementing this class, this must be set by train()
        self._model = None
        # When self._model is a matrix:
        #  - First coordinate indexes the target word
        #  - Second coordinate indexes context word

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    @property
    def name(self) -> str:
        return f"{self.model_type.name} ({self.corpus_meta.name}), r={self.window_radius}"

    @property
    def _model_filename(self):
        return f"{self.corpus_meta.name}_r={self.window_radius}_{self.model_type.slug}"

    @abstractmethod
    def vector_for_word(self, word: str):
        """
        Returns the vector representation of a word.
        :raises WordNotFoundError
        """
        raise NotImplementedError()

    @abstractmethod
    def nearest_neighbours(self, word: str, distance_type: DistanceType, n: int):
        """
        Finds the nearest neighbours to a word.
        """
        raise NotImplementedError()

    def nearest_neighbour(self, word: str, distance_type: DistanceType):
        """
        Finds the nearest neighbour to a word.
        :param word:
        :param distance_type:
        :return:
        """
        return self.nearest_neighbours(word, distance_type, 1)[0]

    def distance_between(self, word_1, word_2, distance_type: DistanceType, truncate_vectors_at_length: int = None):
        """
        Returns the distance between the two specified words
        :param word_1:
        :param word_2:
        :param distance_type:
        :param truncate_vectors_at_length:
        :return:
        :raises: WordNotFoundError
        """
        v_1 = self.vector_for_word(word_1)
        v_2 = self.vector_for_word(word_2)

        # TODO: The vectors that come out of word2vec may not be like this, in which case this won't work.
        # TODO: Verify!
        if truncate_vectors_at_length is not None and truncate_vectors_at_length < v_1.shape[1]:
            v_1 = v_1[:, :truncate_vectors_at_length]
            v_2 = v_2[:, :truncate_vectors_at_length]

        return distance(v_1, v_2, distance_type)


class ScalarSemanticModel(DistributionalSemanticModel, metaclass=ABCMeta):
    """
    A language model where each word is associated with a scalar value.
    """

    def __init__(self,
                 model_type: DistributionalSemanticModel.ModelType,
                 corpus_meta: CorpusMetadata,
                 window_radius: int):
        super().__init__(model_type, corpus_meta)
        self.window_radius = window_radius

        # When implementing this class, this must be set by retrain()
        self._model: numpy.ndarray = None

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    @property
    def name(self) -> str:
        return f"{self.model_type.name} ({self.corpus_meta.name}), r={self.window_radius}"

    @property
    def _model_filename(self):
        return f"{self.corpus_meta.name}_r={self.window_radius}_{self.model_type.name}"

    @abstractmethod
    def scalar_for_word(self, word: str):
        """
        Returns the scalar value for a word.
        """
        raise NotImplementedError()
