"""
===========================
Dealing with multi-word tokens.
===========================

Dr. Cai Wingfield
---------------------------
Embodied Cognition Lab
Department of Psychology
University of Lancaster
c.wingfield@lancaster.ac.uk
caiwingfield.net
---------------------------
2019
---------------------------
"""

from enum import Enum, auto
from functools import reduce
from typing import List


def count_gram(ngram: str) -> int:
    """Returns n where the input is an n-gram."""
    n_spaces = len([pos for pos, char in enumerate(ngram) if char == " "])
    # A 2-gram has 1 space, etc.
    return n_spaces + 1


def bigram_to_hyphenated(bigram: str) -> str:
    """Converts 'hack saw' to 'hack-saw'."""
    return bigram.replace(" ", "-")


def bigram_to_compound(bigram: str) -> str:
    """Converts 'hack saw' to 'hacksaw'."""
    return bigram.replace(" ", "")


def is_unigram(canidate: str) -> bool:
    """True if the candidate is a unigram, else false."""
    return count_gram(canidate) == 1


def is_bigram(candidate: str) -> bool:
    """True if the candidate is a bigram, else false."""
    return count_gram(candidate) == 2


def ngram_to_unigrams(ngram: str) -> List[str]:
    """Converts a ngram to a list of component unigrams."""
    return ngram.strip().split()


class VectorCombinatorType(Enum):
    """An way of combining vectors for multiple words."""

    additive = auto()
    multiplicative = auto()
    mean = auto()
    none = auto()

    @property
    def name(self):
        if self is self.additive:
            return "additive"
        elif self is self.multiplicative:
            return "multiplicative"
        elif self is self.mean:
            return "mean"
        elif self is self.none:
            return "none"
        else:
            raise ValueError()


def multiword_combinator(combinator_type: VectorCombinatorType, *vectors):
    """
    Implementation for each way of combining vectors .
    :param combinator_type:
        The type of combination to perform.
    :param vectors:
        argument list of vectors to combine.
    :return:
    """
    if combinator_type is VectorCombinatorType.additive:
        # As + is associative we can use binary reduction here
        return reduce(lambda v, w: v + w, vectors)
    elif combinator_type is VectorCombinatorType.multiplicative:
        # As * is associative we can use binary reduction here
        return reduce(lambda v, w: v * w, vectors)
    elif combinator_type is VectorCombinatorType.mean:
        # Binary mean is not associative, so we have to add first and then divide.
        return multiword_combinator(VectorCombinatorType.additive, *vectors) / len(vectors)
    elif combinator_type is VectorCombinatorType.none:
        return NotImplementedError()
    else:
        raise ValueError()
