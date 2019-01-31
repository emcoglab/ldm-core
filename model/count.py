"""
===========================
Count models.
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
import sys

from abc import abstractmethod, ABCMeta
from operator import itemgetter
from os import path, makedirs

import numpy
import scipy.sparse

from .base import VectorSemanticModel, DistributionalSemanticModel, ScalarSemanticModel
from ..corpus.corpus import CorpusMetadata, WindowedCorpus
from ..corpus.indexing import FreqDist, TokenIndex
from ..utils.constants import Chirality
from ..utils.exceptions import WordNotFoundError
from ..utils.maths import DistanceType, distance

logger = logging.getLogger(__name__)


class CountVectorModel(VectorSemanticModel):
    """
    A model where vectors are computed by counting contexts.
    """

    def __init__(self,
                 model_type: DistributionalSemanticModel.ModelType,
                 corpus_meta: CorpusMetadata,
                 window_radius: int,
                 freq_dist: FreqDist):
        super().__init__(model_type, corpus_meta, window_radius)
        self.freq_dist: FreqDist = freq_dist
        self.token_index: TokenIndex = TokenIndex.from_freqdist_ranks(freq_dist)

    @property
    def matrix(self) -> scipy.sparse.csr_matrix:
        return self._model

    @property
    def _model_ext(self) -> str:
        return ".npz"

    @abstractmethod
    def _retrain(self):
        raise NotImplementedError()

    def _save(self):
        # Only save a model if we got one.
        assert self.is_trained
        if not path.isdir(self.save_dir):
            logger.warning(f"{self.save_dir} does not exist, making it.")
            makedirs(self.save_dir)
        scipy.sparse.save_npz(path.join(self.save_dir, self._model_filename_with_ext), self._model, compressed=False)

    # BUG: This appears to leak memory if used repeatedly :[
    def _load(self, memory_map: bool = False):

        # Use scipy.sparse.csr_matrix for trained models
        self._model = self._load_npz_with_mmap(
            file=path.join(self.save_dir, self._model_filename_with_ext),
            memory_map=memory_map
        ).tocsr()

        # Make sure nothing's gone wrong
        assert self.is_trained

    @staticmethod
    def _load_npz_with_mmap(file, memory_map: bool = False):
        """
        Copied from scipy.sparse.load_npz, except it allows memory mapping.
        """
        if memory_map:
            with numpy.load(file=file,
                            mmap_mode="r" if memory_map else None,
                            # Works only with Numpy version >= 1.10.0
                            allow_pickle=False
                            ) as loaded:
                try:
                    matrix_format = loaded['format']
                except KeyError:
                    raise ValueError('The file {} does not contain a sparse matrix.'.format(file))

                matrix_format = matrix_format.item()

                if sys.version_info[0] >= 3 and not isinstance(matrix_format, str):
                    # Play safe with Python 2 vs 3 backward compatibility;
                    # files saved with Scipy < 1.0.0 may contain unicode or bytes.
                    matrix_format = matrix_format.decode('ascii')

                try:
                    cls = getattr(scipy.sparse, '{}_matrix'.format(matrix_format))
                except AttributeError:
                    raise ValueError('Unknown matrix format "{}"'.format(matrix_format))

                if matrix_format in ('csc', 'csr', 'bsr'):
                    return cls((loaded['data'], loaded['indices'], loaded['indptr']), shape=loaded['shape'])
                elif matrix_format == 'dia':
                    return cls((loaded['data'], loaded['offsets']), shape=loaded['shape'])
                elif matrix_format == 'coo':
                    return cls((loaded['data'], (loaded['row'], loaded['col'])), shape=loaded['shape'])
                else:
                    raise NotImplementedError('Load is not implemented for '
                                              'sparse matrix of format {}.'.format(matrix_format))
        else:
            # Use standard load function
            return scipy.sparse.load_npz(file)

    def vector_for_id(self, word_id: int):
        """
        Returns the vector representation of a word, given by its index in the corpus.
        :param word_id:
        :return:
        """
        # The first coordinate indexes target words, the second indexes context words.
        # So this should return a vector for the target word whose entries are indexed by context words.
        return self._model[word_id].todense()

    def vector_for_word(self, word: str):
        try:
            word_id = self.token_index.token2id[word]
            return self.vector_for_id(word_id)
        except KeyError:
            raise WordNotFoundError(f"The word '{word}' was not found.")

    def nearest_neighbours(self, word: str, distance_type: DistanceType, n: int, only_consider_most_frequent: int = None):

        if only_consider_most_frequent is not None:
            if only_consider_most_frequent < 1:
                raise ValueError("only_consider_most_frequent must be at least 1")
            vocab_size = only_consider_most_frequent
        else:
            vocab_size = len(self.token_index.id2token)

        if not self.contains_word(word):
            raise WordNotFoundError(f"The word '{word}' was not found.")

        target_id = self.token_index.token2id[word]
        target_vector = self.vector_for_id(target_id)

        nearest_neighbours = []

        for candidate_id in range(0, vocab_size):

            # Skip target word
            if candidate_id == target_id:
                continue

            candidate_vector = self.vector_for_id(candidate_id)
            distance_to_target = distance(candidate_vector, target_vector, distance_type)

            # Add it to the shortlist
            nearest_neighbours.append((candidate_id, distance_to_target))

            # If the list is overfull, remove the lowest one
            if len(nearest_neighbours) > n:
                nearest_neighbours.sort(
                    # Sort by distance, which is the second item in the tuple.
                    key=itemgetter(1),
                    # Sort ascending so the first element is the most similar
                    reverse=False)
                del nearest_neighbours[-1]

            if candidate_id % 10_000 == 0 and candidate_id > 0:
                logger.info(f'\t{candidate_id:,} out of {vocab_size:,} candidates considered. '
                            f'"{self.token_index.id2token[nearest_neighbours[0][0]]}" currently the fave')

        return [self.token_index.id2token[i] for i, dist in nearest_neighbours]

    def contains_word(self, word: str) -> bool:
        if word.lower() in [token.lower() for token in self.token_index.token2id]:
            return True
        else:
            return False


class CountScalarModel(ScalarSemanticModel, metaclass=ABCMeta):
    """
    A context-counting language model where each word is associated with a scalar value.
    """

    def __init__(self,
                 model_type: DistributionalSemanticModel.ModelType,
                 corpus_meta: CorpusMetadata,
                 window_radius: int,
                 freq_dist: FreqDist):
        super().__init__(model_type, corpus_meta, window_radius)
        self.freq_dist: FreqDist = freq_dist
        self.token_index: TokenIndex = TokenIndex.from_freqdist_ranks(freq_dist)

    @property
    def _model_ext(self):
        return ".npz"

    @property
    def vector(self) -> numpy.ndarray:
        return self._model

    def _save(self):
        assert self.is_trained
        if not path.isdir(self.save_dir):
            logger.warning(f"{self.save_dir} does not exist, making it.")
            makedirs(self.save_dir)
        # Can't use scipy save_npz, as this isn's a sparse matrix, it's a vector.
        # So just use numpy savez
        #     https://stackoverflow.com/questions/31468117/python-3-can-pickle-handle-byte-objects-larger-than-4gb
        #     https://github.com/numpy/numpy/issues/3858
        numpy.savez(path.join(self.save_dir, self._model_filename_with_ext), self._model)

    def _load(self, memory_map: bool = False):

        if memory_map:
            logger.warning(f"Memory mapping not currently supported for Vector models")

        self._model = numpy.load(path.join(self.save_dir, self._model_filename_with_ext))["arr_0"]
        assert self.is_trained

    def scalar_for_word(self, word: str):

        if not self.contains_word(word):
            raise WordNotFoundError(f"The word '{word}' was not found.")

        return self._model[self.token_index.token2id[word]]

    def contains_word(self, word: str) -> bool:
        if word.lower() in self.token_index.token2id:
            return True
        else:
            return False


class UnsummedCoOccurrenceCountModel(CountVectorModel):
    """
    A model where vectors consist of context counts at a fixed distance either on the left or right of a window.
    """

    def __init__(self,
                 corpus_meta: CorpusMetadata,
                 window_radius: int,
                 freq_dist: FreqDist,
                 chirality: Chirality):
        super().__init__(DistributionalSemanticModel.ModelType.unsummed_cooccurrence,
                         corpus_meta, window_radius, freq_dist)
        self._chirality = chirality

    # Overwrite, to include chirality
    @property
    def name(self) -> str:
        return f"{self.model_type.name} ({self.corpus_meta.name}), r={self.window_radius}, {self._chirality.name}"

    # Overwrite, to include chirality
    @property
    def _model_filename(self):
        return f"{self.corpus_meta.name}_r={self.window_radius}_{self.model_type.slug}_{self._chirality}"

    def _retrain(self):

        vocab_size = len(self.freq_dist)

        # Initialise cooccurrence matrices

        # We will store left- and right-cooccurrences separately.
        # At this stage we define radius-n cooccurrence to be words which are /exactly/ n words apart,
        # rather than /up-to/ n words apart.
        # This will greatly speed up computation, and we can sum the values later much faster to get the
        # standard "summed" n-gram counts.

        # First coordinate points to target word
        # Second coordinate points to context word
        # Use scipy.sparse.lil_matrix for direct indexed access
        self._model = scipy.sparse.lil_matrix((vocab_size, vocab_size))

        # We will produce a window which contains EITHER the left or right context, plus the target word (+1)
        window_size = self.window_radius + 1

        # Start scanning the corpus
        window_count = 0
        for window in WindowedCorpus(self.corpus_meta, window_size):

            if self._chirality is Chirality.left:
                # For a left-hand context, the target token is on the far right
                # And the context token is on the far left
                target_index = -1
                context_index = 0
            elif self._chirality is Chirality.right:
                # For a right-hand context, the target token is on the far left
                # And the context token is on the far right
                target_index = 0
                context_index = -1
            else:
                raise ValueError()

            target_token = window[target_index]
            context_token = window[context_index]

            target_id = self.token_index.token2id[target_token]
            context_id = self.token_index.token2id[context_token]

            # TODO: Left- and right-context matrices are transposes of one another.  For the edge-most elements of
            # TODO: every window, one is either the target or the context, and the other is the other.  So we could
            # TODO: speed up this whole shebang by only computing and saving the lower-triangular elements.
            self._model[target_id, context_id] += 1

            # Count cooccurrences
            window_count += 1

            if window_count % 1_000_000 == 0:
                logger.info(f"\t{window_count:,} tokens processed")

        # Using csr for trained models
        self._model = self._model.tocsr()


class CoOccurrenceCountModel(CountVectorModel):
    """
    A model where vectors consist of the counts of context words within a window.

    n(c,t)

    c: context token
    t: target token
    """

    def __init__(self,
                 corpus_meta: CorpusMetadata,
                 window_radius: int,
                 freq_dist: FreqDist):
        super().__init__(VectorSemanticModel.ModelType.cooccurrence, corpus_meta, window_radius, freq_dist)

    def _retrain(self):

        vocab_size = len(self.freq_dist)

        # Start with an empty sparse matrix
        self._model = scipy.sparse.csr_matrix((vocab_size, vocab_size))

        # We load the unsummed cooccurrence matrices in in sequence, and accumulate them to save the summed
        for radius in range(1, self.window_radius + 1):
            # Accumulate both left and right occurrences
            for chirality in Chirality:
                # Get each unsummed model
                unsummed_model = UnsummedCoOccurrenceCountModel(self.corpus_meta, radius, self.freq_dist, chirality)
                unsummed_model.train()

                # And add it to the current matrix
                self._model += unsummed_model.matrix

                # Prompt GC
                del unsummed_model


class LogCoOccurrenceCountModel(CountVectorModel):
    """
    A model where vectors consist of the log of context counts within a window.
    Uses the log (n+1) method to account for 0-and-1-frequency co-occurrences.

    log [ n(c,t) + 1 ]

    c: context token
    t: target token
    """

    def __init__(self,
                 corpus_meta: CorpusMetadata,
                 window_radius: int,
                 freq_dist: FreqDist):
        super().__init__(VectorSemanticModel.ModelType.log_cooccurrence, corpus_meta, window_radius, freq_dist)

    def _retrain(self):
        # Get the cooccurrence model
        ngram_model = CoOccurrenceCountModel(self.corpus_meta, self.window_radius, self.freq_dist)
        ngram_model.train()

        self._model = ngram_model.matrix
        del ngram_model
        # Apply log to entries in the cooccurrence matrix
        self._model.data = numpy.log10(self._model.data + 1)
        self._model.eliminate_zeros()


class CoOccurrenceProbabilityModel(CountVectorModel):
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
                 window_radius: int,
                 freq_dist: FreqDist):
        super().__init__(VectorSemanticModel.ModelType.cooccurrence_probability,
                         corpus_meta, window_radius, freq_dist)

    def _retrain(self):
        # Get the cooccurrence model
        ngram_model = CoOccurrenceCountModel(self.corpus_meta, self.window_radius, self.freq_dist)
        ngram_model.train()

        # The probability is just the cooccurrence count, divided by the width of the window and the size of the corpus
        self._model = ngram_model.matrix
        del ngram_model
        # The width of the window is twice the radius
        # We don't do 2r+1 because we only count the context words, not the target word
        self._model /= self.window_radius * 2
        self._model /= self.freq_dist.N()


class TokenProbabilityModel(CountScalarModel):
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
                 window_radius: int,
                 freq_dist: FreqDist):
        super().__init__(VectorSemanticModel.ModelType.token_probability,
                         corpus_meta, window_radius, freq_dist)

    def _retrain(self):
        # Get the cooccurrence model
        ngram_model = CoOccurrenceCountModel(self.corpus_meta, self.window_radius, self.freq_dist)
        ngram_model.train()

        # The probability is just the token count, divided by the width of the window and the size of the corpus
        # We're summing over contexts (second dim) to get a count of the targets
        self._model = numpy.sum(ngram_model.matrix, 1)
        del ngram_model
        # The width of the window is twice the radius
        # We don't do 2r+1 because we only count the context words, not the target word
        self._model /= self.window_radius * 2
        self._model /= self.freq_dist.N()


class ConditionalProbabilityModel(CountVectorModel):
    """
    A model where vectors consist of n-gram counts normalised by token probabilities.

    p(c|t) = p(c,t) / p(t)

    c: context token
    t: target token
    """

    def __init__(self,
                 corpus_meta: CorpusMetadata,
                 window_radius: int,
                 freq_dist: FreqDist):
        super().__init__(VectorSemanticModel.ModelType.conditional_probability, corpus_meta, window_radius, freq_dist)

    def _retrain(self):
        ngram_probability_model = CoOccurrenceProbabilityModel(self.corpus_meta, self.window_radius, self.freq_dist)
        ngram_probability_model.train()

        # Convert to csr for linear algebra
        self._model = ngram_probability_model.matrix
        del ngram_probability_model

        token_probability_model = TokenProbabilityModel(self.corpus_meta, self.window_radius, self.freq_dist)
        token_probability_model.train()

        # Here we divide each n-gram probability value by the token probability value.
        # This amounts to dividing each 0th-dim-slice of the matrix by a single value
        #
        #                                  p(c,t)         p(t)
        #
        #                               [ [-, -, -] ,     [ - ,     <- entire mx row to be div'd by this vec entry
        # mx indexed by t on 0th dim ->   [-, -, -] ,  /    - , <- vec indexed by t on 0th dim
        #                                 [-, -, -] ]       - ]
        #                                     ^
        #                                     |
        #                                     mx indexed by c on 1st dim
        #
        # According to https://stackoverflow.com/a/12238133/2883198, this is how you do that:
        self._model.data = self._model.data / token_probability_model.vector.repeat(numpy.diff(self._model.indptr))
        # The division causes the data to become a 1-d numpy.matrix, so we convert it back into a numpy.ndarray
        self._model.data = numpy.squeeze(numpy.asarray(self._model.data))
        self._model.eliminate_zeros()


class ContextProbabilityModel(CountScalarModel):
    """
    A model where scalars consist of the probability that any token is the target.

    p(c) = Sum_t n(c,t) / NW

    c: context token
    t: target token
    N: size of corpus
    W: width of window
    """

    def __init__(self,
                 corpus_meta: CorpusMetadata,
                 window_radius: int,
                 freq_dist: FreqDist):
        super().__init__(VectorSemanticModel.ModelType.context_probability,
                         corpus_meta, window_radius, freq_dist)

    def _retrain(self):
        # Get the cooccurrence model
        ngram_model = CoOccurrenceCountModel(self.corpus_meta, self.window_radius, self.freq_dist)
        ngram_model.train()

        # The probability is just the token count, divided by the width of the window and the size of the corpus
        # We're summing over targets (first dim) to get the count of the contexts
        self._model = numpy.sum(ngram_model.matrix, 0)
        # The width of the window is twice the radius
        # We don't do 2r+1 because we only count the context words, not the target word
        self._model /= self.window_radius * 2
        self._model /= self.freq_dist.N()


class ProbabilityRatioModel(CountVectorModel):
    """
    A model where vectors consist of the ratio of probabilities.

    r(c,t) = p(c|t) / p(c)

    c: context token
    t: target token
    """

    def __init__(self,
                 corpus_meta: CorpusMetadata,
                 window_radius: int,
                 freq_dist: FreqDist):
        super().__init__(DistributionalSemanticModel.ModelType.probability_ratio, corpus_meta, window_radius, freq_dist)

    def _retrain(self):
        cond_prob_model = ConditionalProbabilityModel(self.corpus_meta, self.window_radius, self.freq_dist)
        cond_prob_model.train()

        # Convert to csr for linear algebra
        self._model = cond_prob_model.matrix
        del cond_prob_model

        context_probability_model = ContextProbabilityModel(self.corpus_meta, self.window_radius, self.freq_dist)
        context_probability_model.train()

        # Here we divide each conditional n-gram probability value by the context probability value.
        # This amounts to dividing each 0th-dim-slice of the matrix by a single value
        #
        #                                      mx indexed by c on 1st dim
        #                                      |
        #                                      v
        #                               [ [ -, -, - ] ,
        # mx indexed by t on 0th dim ->   [ -, -, - ] ,   p(c|t)
        #                                 [ -, -, - ] ]
        #                                      /
        #                                 [ -, -, - ]     p(c)
        #                                   ^     ^
        #                                   |     |
        #                                   |     vec indexed by c on 0th dim
        #                                   |
        #                                   entire mx col to be div'd by this vec entry
        #
        # We follow the same method as for the ConditionalProbabilityModel, but that's for dividing each row by a
        # corresponding vector element, and we want to divide each column by the corresponding vector element.  We know
        # that the row method is fast, so we'll transpose, divide, transpose back.
        self._model = self._model.transpose().tocsr()
        self._model.data = self._model.data / context_probability_model.vector.repeat(numpy.diff(self._model.indptr))
        # The division causes the data to become a 1-d numpy.matrix, so we convert it back into a numpy.ndarray
        self._model.data = numpy.squeeze(numpy.asarray(self._model.data))
        self._model.eliminate_zeros()
        self._model = self._model.transpose().tocsr()


class PMIModel(CountVectorModel):
    """
    A model where the vectors consist of the pointwise mutual information between the context and the target.

     PMI(c,t) = log_2 r(c,t)

    c: context token
    t: target token
    """

    def __init__(self,
                 corpus_meta: CorpusMetadata,
                 window_radius: int,
                 freq_dist: FreqDist):
        super().__init__(DistributionalSemanticModel.ModelType.pmi, corpus_meta, window_radius, freq_dist)

    def _retrain(self):

        # Start with probability ratio model
        ratios_model = ProbabilityRatioModel(self.corpus_meta, self.window_radius, self.freq_dist)
        ratios_model.train()

        # Copy ratios model matrix
        self._model = ratios_model.matrix
        del ratios_model

        # PMI model data is log_2 of ratios model data
        self._model.data = numpy.log2(self._model.data)
        self._model.eliminate_zeros()


class PPMIModel(CountVectorModel):
    """
    A model where the vectors consist of the positive pointwise mutual information between the context and the target.

    PMI^+(c,t) = max(0, PMI(c,t))

    where: PMI(c,t) = log_2 r(c,t)

    c: context token
    t: target token
    """

    def __init__(self,
                 corpus_meta: CorpusMetadata,
                 window_radius: int,
                 freq_dist: FreqDist):
        super().__init__(DistributionalSemanticModel.ModelType.ppmi, corpus_meta, window_radius, freq_dist)

    def _retrain(self):

        # Start with pmi
        pmi_model = PMIModel(self.corpus_meta, self.window_radius, self.freq_dist)
        pmi_model.train()

        # Copy pmi model matrix
        self._model = pmi_model.matrix
        del pmi_model

        # Keep non-negative values only
        self._model.data[self._model.data < 0] = 0
        self._model.eliminate_zeros()
