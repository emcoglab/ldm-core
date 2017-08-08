import logging
import os
from abc import abstractmethod

import numpy as np
import scipy.io as sio
import scipy.sparse as sps

from ..corpus.corpus import CorpusMetadata, WindowedCorpus
from ..model.base import VectorSpaceModel, ScalarModel, LanguageModel
from ..utils.constants import Chirality
from ..utils.indexing import TokenIndexDictionary
from ..utils.maths import Distance

logger = logging.getLogger(__name__)


class CountModel(VectorSpaceModel):
    """
    A model where vectors are computed by counting contexts/
    """
    def __init__(self,
                 model_type: LanguageModel.ModelType,
                 corpus_meta: CorpusMetadata,
                 save_dir: str,
                 window_radius: int,
                 token_indices: TokenIndexDictionary):
        super().__init__(model_type, corpus_meta, save_dir, window_radius)
        self.token_indices = token_indices

    @property
    def matrix(self):
        return self._model

    @abstractmethod
    def train(self, force_retrain: bool = False):
        raise NotImplementedError()

    def save(self):
        logger.info(f"Saving cooccurrence matrix")
        sio.mmwrite(os.path.join(self.save_dir, self._model_filename), self._model)

    def load(self):
        logger.info(f"Loading {self.corpus_meta.name} corpus, radius {self.window_radius}")
        self._model = sio.mmread(os.path.join(self.save_dir, self._model_filename))

    def vector_for_id(self, word_id: int):
        """
        Returns the vector representation of a word, given by its index in the corpus.
        :param word_id:
        :return:
        """
        return self._model[word_id]

    def vector_for_word(self, word: str):
        word_id = self.token_indices.token2id(word)
        return self.vector_for_id(word_id)

    def nearest_neighbours(self, word: str, distance_type: Distance.Type, n: int):

        vocab_size = len(self.token_indices)

        target_id = self.token_indices.token2id(word)
        target_vector = self.vector_for_id(target_id)

        nearest_neighbours = []

        for candidate_id in range(0, vocab_size):

            # Skip target word
            if candidate_id == target_id:
                continue

            candidate_vector = self.vector_for_id(candidate_id)
            distance_to_target = Distance.d(candidate_vector, target_vector, distance_type)

            # Add it to the shortlist
            nearest_neighbours.append((candidate_id, distance_to_target))
            nearest_neighbours.sort(key=lambda c_id, c_dist: c_dist)

            # If the list is overfull, remove the lowest one
            if len(nearest_neighbours) > n:
                nearest_neighbours = nearest_neighbours[:-1]

        return [self.token_indices.id2token(i) for i, dist in nearest_neighbours]


class UnsummedNgramCountModel(CountModel):
    """
    A model where vectors consist of context counts at a fixed distance either on the left or right of a window.
    """

    def __init__(self,
                 corpus_meta: CorpusMetadata,
                 save_dir: str,
                 window_radius: int,
                 token_indices: TokenIndexDictionary,
                 chirality: Chirality):
        super().__init__(LanguageModel.ModelType.ngram_unsummed,
                         corpus_meta, save_dir, window_radius, token_indices)
        self._chirality = chirality

        # Overwrite, to include chirality
        self._matrix_filename = f"{self.corpus_meta.name}_r={self.window_radius}_{self.model_type}_{self._chirality}"

    def train(self, force_retrain: bool = False):

        vocab_size = len(self.token_indices)

        # Skip ones which are already done
        if os.path.isfile(self._matrix_filename) and not force_retrain:
            logger.info(f"Loading {self.corpus_meta.name} corpus, radius {self.window_radius}")
            self._model = sio.mmread(self._matrix_filename)
        else:
            logger.info(f"Working on {self.corpus_meta.name} corpus, radius {self.window_radius}")

            # Initialise cooccurrence matrices

            # We will store left- and right-cooccurrences separately.
            # At this stage we define radius-n cooccurrence to be words which are /exactly/ n words apart,
            # rather than /up-to/ n words apart.
            # This will greatly speed up computation, and we can sum the values later much faster to get the
            # standard "summed" n-gram counts.

            # First coordinate points to target word
            # Second coordinate points to context word
            self._model = sps.lil_matrix((vocab_size, vocab_size))

            # Start scanning the corpus
            window_count = 0
            for window in WindowedCorpus(self.corpus_meta, self.window_radius):

                # The target token is the one in the middle, whose index is the radius of the window
                target_token = window[self.window_radius]
                target_id = self.token_indices.token2id[target_token]

                if self._chirality is Chirality.left:
                    # For the left occurrences, we look in the first position in the window; index 0
                    lr_index = 0
                elif self._chirality is Chirality.right:
                    # For the right occurrences, we look in the last position in the window; index -1
                    lr_index = -1
                else:
                    raise ValueError()

                # Count lh occurrences
                context_token = window[lr_index]
                context_id = self.token_indices.token2id[context_token]
                self._model[target_id, context_id] += 1

                window_count += 1
                if window_count % 1_000_000 == 0:
                    logger.info(f"\t{window_count:,} tokens processed")


class NgramCountModel(CountModel):
    """
    A model where vectors consist of the counts of context words within a window.

    n(c,t)

    c: context token
    t: target token
    """

    def __init__(self,
                 corpus_meta: CorpusMetadata,
                 save_dir: str,
                 window_radius: int,
                 token_indices: TokenIndexDictionary):
        super().__init__(VectorSpaceModel.ModelType.ngram, corpus_meta, save_dir, window_radius, token_indices)

    def train(self, force_retrain: bool = False):

        vocab_size = len(self.token_indices)
        self._model = sps.lil_matrix((vocab_size, vocab_size))

        if os.path.isfile(self._model_filename) and not force_retrain:
            self.load()
        else:
            # We load the unsummed cooccurrence matrices in in sequence, and accumulate them to save the summed
            for radius in range(1, max(self.window_radius) + 1):
                # Accumulate both left and right occurrences
                for chirality in Chirality:
                    # Load each unsummed model
                    unsummed_model = UnsummedNgramCountModel(self.corpus_meta, self.save_dir, self.window_radius,
                                                             self.token_indices,
                                                             chirality)
                    unsummed_model.load()

                    # And add it to the current matrix
                    self._model += unsummed_model.matrix


class LogNgramModel(CountModel):
    """
    A model where vectors consist of the log of context counts within a window.

    log n(c,t)

    c: context token
    t: target token
    """

    def __init__(self,
                 corpus_meta: CorpusMetadata,
                 save_dir: str,
                 window_radius: int,
                 token_indices: TokenIndexDictionary):
        super().__init__(VectorSpaceModel.ModelType.log_ngram,
                         corpus_meta, save_dir, window_radius, token_indices)

    def train(self, force_retrain: bool = False):

        vocab_size = len(self.token_indices)
        self._model = sps.lil_matrix((vocab_size, vocab_size))

        if os.path.isfile(self._model_filename) and not force_retrain:
            self.load()
        else:

            # Load the ngram model
            ngram_model = NgramCountModel(self.corpus_meta, self.save_dir, self.window_radius, self.token_indices)
            ngram_model.load()

            # Apply log to entries in the ngram matrix
            self._model = ngram_model.matrix
            self._model.data = np.log10(self._model.data)


class NgramProbabilityModel(CountModel):
    """
    A model where vectors consist of the probability that a given context is found within a window around the target.

    p(c,t) = n(c,t) / NW

    c: context token
    t: target token
    N: size of corpus
    W: width of window
    """

    def __init__(self,
                 corpus_meta: CorpusMetadata,
                 save_dir: str,
                 window_radius: int,
                 token_indices: TokenIndexDictionary):
        super().__init__(VectorSpaceModel.ModelType.ngram_probability,
                         corpus_meta, save_dir, window_radius, token_indices)

    def train(self, force_retrain: bool = False):

        vocab_size = len(self.token_indices)
        self._model = sps.lil_matrix((vocab_size, vocab_size))

        if os.path.isfile(self._model_filename) and not force_retrain:
            self.load()
        else:

            # Load the ngram model
            ngram_model = NgramCountModel(self.corpus_meta, self.save_dir, self.window_radius, self.token_indices)
            ngram_model.load()

            # The probability is just the ngram count, divided by the width of the window and the size of the corpus
            self._model = ngram_model.matrix
            # The width of the window is twice the radius
            # We don't do 2r+1 because we only count the context words, not the target word
            self._model /= self.window_radius * 2
            # TODO: get the corpus size from somewhere... maybe a freqdist?
            self._model /= corpus_size


class TokenProbabilityModel(ScalarModel):
    """
    A model where ~vectors~ consist of the probability that any token is the target.

    p(t) = Sum_c n(c,t) / NW

    c: context token
    t: target token
    N: size of corpus
    W: width of window
    """

    def __init__(self,
                 corpus_meta: CorpusMetadata,
                 save_dir: str,
                 window_radius: int,
                 token_indices: TokenIndexDictionary):
        super().__init__(VectorSpaceModel.ModelType.token_probability,
                         corpus_meta, save_dir, window_radius, token_indices)

    def train(self, force_retrain: bool = False):

        if os.path.isfile(self._model_filename) and not force_retrain:
            self.load()
        else:

            # Load the ngram model
            ngram_model = NgramCountModel(self.corpus_meta, self.save_dir, self.window_radius, self.token_indices)
            ngram_model.load()

            # The probability is just the ngram count, divided by the width of the window and the size of the corpus
            # TODO: this is the wrong initialisation.
            # TODO: sum in here.
            self._model = ngram_model.matrix
            # The width of the window is twice the radius
            # We don't do 2r+1 because we only count the context words, not the target word
            self._model /= self.window_radius * 2
            # TODO: get the corpus size from somewhere... maybe a freqdist?
            self._model /= corpus_size

    def scalar_for_word(self, word: str):
        return self._model[self.token_indices.token2id[word]]

    def save(self):
        logger.info(f"Saving cooccurrence matrix")
        sio.mmwrite(os.path.join(self.save_dir, self._model_filename), self._model)

    def load(self):
        logger.info(f"Loading {self.corpus_meta.name} corpus, radius {self.window_radius}")
        self._model = sio.mmread(os.path.join(self.save_dir, self._model_filename))


class ConditionalProbabilityModel(CountModel):
    """
    A model where vectors consist of n-gram counts normalised by token probabilities.

    p(c|t) = p(c,t) / p(t)

    c: context token
    t: target token
    """

    def __init__(self,
                 corpus_meta: CorpusMetadata,
                 save_dir: str,
                 window_radius: int,
                 token_indices: TokenIndexDictionary):
        super().__init__(VectorSpaceModel.ModelType.conditional_probability,
                         corpus_meta, save_dir, window_radius, token_indices)

    def train(self, force_retrain: bool = False):
        # TODO
        pass


class ContextProbabilityModel(ScalarModel):
    """
    A model where ~vectors~ consist of the probability that any token is the target.

    p(c) = Sum_t n(c,t) / NW

    c: context token
    t: target token
    N: size of corpus
    W: width of window
    """

    def __init__(self,
                 corpus_meta: CorpusMetadata,
                 save_dir: str,
                 window_radius: int,
                 token_indices: TokenIndexDictionary):
        super().__init__(VectorSpaceModel.ModelType.context_probability,
                         corpus_meta, save_dir, window_radius, token_indices)

    def scalar_for_word(self, word: str):
        return self._model[self.token_indices.token2id[word]]

    def save(self):
        logger.info(f"Saving cooccurrence matrix")
        sio.mmwrite(os.path.join(self.save_dir, self._model_filename), self._model)

    def load(self):
        logger.info(f"Loading {self.corpus_meta.name} corpus, radius {self.window_radius}")
        self._model = sio.mmread(os.path.join(self.save_dir, self._model_filename))

    def train(self, force_retrain: bool = False):

        if os.path.isfile(self._model_filename) and not force_retrain:
            self.load()
        else:

            # Load the ngram model
            ngram_model = NgramCountModel(self.corpus_meta, self.save_dir, self.window_radius, self.token_indices)
            ngram_model.load()

            # The probability is just the ngram count, divided by the width of the window and the size of the corpus
            # TODO: this is the wrong initialisation.
            # TODO: sum in here.
            self._model = ngram_model.matrix
            # The width of the window is twice the radius
            # We don't do 2r+1 because we only count the context words, not the target word
            self._model /= self.window_radius * 2
            # TODO: get the corpus size from somewhere... maybe a freqdist?
            self._model /= corpus_size


# TODO: is there another, more intuitive "ratio" formula for this?
class ProbabilityRatioModel(CountModel):
    """
    A model where vectors consist of the ratio of probabilities.

    r(c,t) = p(c|t) / p(c)


    c: context token
    t: target token
    """
    # TODO
    pass


class PMIModel(CountModel):
    """
    A model where the vectors consist of the pointwise mutual information between the context and the target.

    PMI(c,t) = log_2 r(c,t)

    c: context token
    t: target token
    """
    # TODO
    pass


class PPMIModel(CountModel):
    """
    A model where the vectors consist of the positive pointwise mutual information between the context and the target.

    PMI^+(c,t) = max(0,PMI(c,t))

    c: context token
    t: target token
    """
    # TODO
    pass