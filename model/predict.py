"""
===========================
Predict models.
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

from abc import abstractmethod
from os import path, makedirs

import gensim

from .base import VectorSemanticModel, DistributionalSemanticModel
from ..utils.exceptions import WordNotFoundError
from ..utils.maths import distance, DistanceType
from ..corpus.corpus import CorpusMetadata, BatchedCorpus

logger = logging.getLogger(__name__)


class PredictVectorModel(VectorSemanticModel):
    """A vector space model where words are predicted rather than counted."""
    def __init__(self,
                 model_type: DistributionalSemanticModel.ModelType,
                 corpus_meta: CorpusMetadata,
                 window_radius: int,
                 embedding_size: int,
                 n_workers: int = None):
        super().__init__(model_type, corpus_meta, window_radius)
        self.embedding_size = embedding_size

        # Recommended value from Mandera et al. (2017).
        # Baroni et al. (2014) recommend either 5 or 10, but 10 tended to perform slightly better overall.
        self._negative_sampling = 10
        # Recommended value from Mandera et al. (2017) and Baroni et al. (2014).
        self._sub_sample = 1e-5

        # Parallel workers
        if n_workers is not None:
            self._workers = n_workers
        else:
            # Default to 8
            self._workers = 8

        self._corpus = BatchedCorpus(corpus_meta, batch_size=1_000)
        self._model: gensim.models.Word2Vec = None

    @property
    def name(self) -> str:
        return f"{self.model_type.name} ({self.corpus_meta.name}), r={self.window_radius}, s={self.embedding_size}"

    @property
    def _model_filename(self):
        # Include embedding size
        return f"{self.corpus_meta.name}_r={self.window_radius}_s={self.embedding_size}_{self.model_type.slug}"

    @property
    def _model_ext(self) -> str:
        # Word2Vec models don't have an extension
        return ""

    def _save(self):
        assert self.is_trained
        if not path.isdir(self.save_dir):
            logger.warning(f"{self.save_dir} does not exist, making it.")
            makedirs(self.save_dir)
        self._model.save(path.join(self.save_dir, self._model_filename))

    def _load(self, memory_map: bool = False):
        self._model = gensim.models.Word2Vec.load(fname=path.join(self.save_dir, self._model_filename),
                                                  mmap="r" if memory_map else None)
        assert self.is_trained

    @abstractmethod
    def _retrain(self):
        raise NotImplementedError()

    def nearest_neighbours_with_distances(self, word: str, distance_type: DistanceType, n: int,
                                          only_consider_most_frequent: int = None):

        if not self.contains_word(word):
            raise WordNotFoundError(f"The word '{word}' was not found.")

        if only_consider_most_frequent is not None and only_consider_most_frequent < 1:
            raise ValueError("only_consider_most_frequent must be at least 1")

        # TODO: This param only works if the vocab is in frequency-sorted order.
        # TODO: Verify that this is true and then remove this warning!
        if only_consider_most_frequent is not None:
            logger.warning("only_consider_most_frequent feature is completely untested and may not work at all!")

        if distance_type is DistanceType.cosine:
            # gensim implements cosine anyway, so this is an easy shortcut
            return self._model.wv.most_similar(positive=word, topn=n, restrict_vocab=only_consider_most_frequent)
        else:
            # Other distances aren't implemented natively
            target_word = word
            target_vector = self.vector_for_word(target_word)

            nearest_neighbours = []

            for candidate_word in self._model.raw_vocab:

                # Skip target word
                if candidate_word is target_word:
                    continue

                candidate_vector = self.vector_for_word(candidate_word)
                distance_to_target = distance(candidate_vector, target_vector, distance_type)

                # Add it to the shortlist
                nearest_neighbours.append((candidate_word, distance_to_target))
                nearest_neighbours.sort(key=lambda w, d: d)

                # If the list is overfull, remove the lowest one
                if len(nearest_neighbours) > n:
                    nearest_neighbours = nearest_neighbours[:-1]

            return nearest_neighbours

    def vector_for_word(self, word: str):
        try:
            return self._model.wv.word_vec(word)
        except KeyError:
            raise WordNotFoundError(f"The word '{word}' was not found.")

    def contains_word(self, word: str) -> bool:
        if word.lower() in self._model.wv.vocab:
            return True
        else:
            return False


class CbowModel(PredictVectorModel):
    """A vector space model trained using CBOW."""
    def __init__(self,
                 corpus_meta: CorpusMetadata,
                 window_radius: int,
                 embedding_size: int,
                 n_workers: int = None):
        super().__init__(VectorSemanticModel.ModelType.cbow,
                         corpus_meta, window_radius, embedding_size, n_workers)

    def _retrain(self):

        self._model = gensim.models.Word2Vec(
            # This is called "sentences", but they all get concatenated, so it doesn't matter.
            sentences=self._corpus,
            # This is CBOW, so don't use Skip-gram
            sg=0,
            size=self.embedding_size,
            window=self.window_radius,
            negative=self._negative_sampling,
            sample=self._sub_sample,
            # If we do filtering of word frequency, we'll do it in the corpus.
            min_count=0,
            workers=self._workers)


class SkipGramModel(PredictVectorModel):
    """A vector space model trained using Skip-gram."""
    def __init__(self,
                 corpus_meta: CorpusMetadata,
                 window_radius: int,
                 embedding_size: int,
                 n_workers: int = None):
        super().__init__(VectorSemanticModel.ModelType.skip_gram,
                         corpus_meta, window_radius, embedding_size, n_workers)

    def _retrain(self):

        self._model = gensim.models.Word2Vec(
            # This is called "sentences", but they all get concatenated, so it doesn't matter.
            sentences=self._corpus,
            # This is Skip-gram, so make sure we use it!
            sg=1,
            size=self.embedding_size,
            window=self.window_radius,
            negative=self._negative_sampling,
            sample=self._sub_sample,
            # If we do filtering of word frequency, we'll do it in the corpus.
            min_count=0,
            workers=self._workers)
