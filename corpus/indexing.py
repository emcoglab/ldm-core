"""
===========================
Converting string tokens to and from int indexes.
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

import json
import os
import logging
from typing import List, Dict, Iterable

import nltk

from .corpus import BatchedCorpus

logger = logging.getLogger(__name__)


class TokenIndex(object):
    """A matching between tokens and ids."""
    def __init__(self, token2id: Dict):
        """
        :param token2id:
            A dictionary with tokens as keys and ids as values.
            ids should be unique
        """
        self.token2id: Dict = token2id
        self.id2token: Dict = dict((v, k) for k, v in token2id.items())

    @property
    def tokens(self):
        return [k for k, v in self.token2id.items()]

    @property
    def indices(self):
        return [k for k, v in self.id2token.items()]

    def export_json(self, filename):
        """
        Saves a json dump of token indices.
        :param filename:
        :return:
        """
        with open(filename, mode="w") as file:
            json.dump(self.token2id, file,
                      # Remove whitespace for smaller files
                      separators=(',', ':'))

    @classmethod
    def from_freqdist_ranks(cls, freq_dist: 'FreqDist'):
        """
        Builds a TokenIndex from a FreqDist.
        Ids for tokens will be taken from the token rank in the FreqDist, 0-indexed so they can be used as array
        indices. So the most frequent token will have id 0, the second-most frequent will have id 1, etc.
        :param freq_dist:
            The FreqDist from which to build the index.
        :return:
        """
        token2id: Dict = dict()
        current_id = 0
        for token, _freq in freq_dist.most_common():
            token2id[token] = current_id
            current_id += 1
        return cls(token2id)


class FreqDist(nltk.probability.FreqDist):
    """
    Extension of nltk.probability.FreqDist.
    """

    _file_extension = ".freqdist"

    @classmethod
    def could_load(cls, filename: str) -> bool:
        if not filename.endswith(FreqDist._file_extension):
            filename += FreqDist._file_extension
        return os.path.isfile(filename)

    def save(self, filename: str):
        """Save the FreqDist to a file."""
        if not filename.endswith(FreqDist._file_extension):
            filename += FreqDist._file_extension
        with open(filename, mode="w") as file:
            # The fundamental data of a FreqDist is stored as a dict (which in fact it inherits from, I think?)
            # We can save it as a json text dump, to make it a little more portable (and also human readable!)
            json.dump(dict(self), file,
                      # Remove whitespace for smaller files
                      separators=(',', ':'))

    @classmethod
    def load(cls, filename: str) -> 'FreqDist':
        """Load the FreqDist from a file."""
        if not filename.endswith(FreqDist._file_extension):
            filename += FreqDist._file_extension
        with open(filename, mode="r") as file:
            # instances are loaded as dicts, so we must cast it up to a FreqDist
            return cls(json.load(file))

    def rank(self, token) -> int:
        """
        The rank of a token in an ordered list of all sampled tokens, ordered most- to least-frequent.

        NOTE: The rank of the most frequent token is 0, so that the rank can be used as an index in a list of ordered
              terms.

        A rank of -1 indicates that the token is not found.
        """
        freq = self[token]
        if freq == 0:
            # If the token is not found, return -1
            return -1
        else:
            rank = self.most_common().index((token, freq))
            return rank

    def most_common_tokens(self, top_n: int = None) -> List[str]:
        """A list of the most common tokens, in order."""
        return list(self.iter_most_common_tokens(top_n))

    def iter_most_common_tokens(self, top_n: int = None) -> Iterable[str]:
        """Iterates the most common tokens, in order."""
        for word, _ in self.most_common(top_n):
            yield word

    @classmethod
    def from_batched_corpus(cls, batched_corpus: BatchedCorpus) -> 'FreqDist':
        # Initially create an empty FreqDist
        freq_dist = cls()
        # Then add to it from each batch of the corpus.
        # Means we don't have to have the whole corpus in memory at once!
        token_count = 0
        for batch in batched_corpus:
            freq_dist += cls(batch)
            token_count += batched_corpus.batch_size
            logger.info(f"\t{token_count:,} tokens counted")
        return freq_dist


class LetterIndexing(object):
    _the_alphabet = list("abcdefghijklmnopqrstuvwxyz")

    @staticmethod
    def letter2int(letter: str) -> int:
        """
        Converts roman letters a, b, c, ... into integers 0, 1, 2, ...
        """
        i = LetterIndexing._the_alphabet.index(letter)
        if i == -1:
            raise KeyError()
        return i

    @staticmethod
    def int2letter(i: int) -> str:
        """
        Converts integers 0, 1, 2, ... into letters a, b, c, ...
        """
        return LetterIndexing._the_alphabet[i]
