"""
===========================
Mathematical things.
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

from enum import Enum, auto

import nltk
import numpy
from numpy import log
from scipy import spatial
from scipy.special import beta as beta_function
from scipy.stats import beta as beta_distribution


class CorrelationType(Enum):
    """
    Representative of a correlation type.
    """
    Pearson  = auto()
    Spearman = auto()

    @property
    def name(self) -> str:
        """
        A string representation of the correlation type.
        """
        if self is CorrelationType.Pearson:
            return "Pearson"
        elif self is CorrelationType.Spearman:
            return "Spearman"
        else:
            raise ValueError()


class DistanceType(Enum):
    """
    Representative of a distance type.
    """
    correlation = auto()
    cosine      = auto()
    Euclidean   = auto()

    @property
    def name(self) -> str:
        """
        A string representation of the distance type.
        """
        if self is DistanceType.Euclidean:
            return "Euclidean"
        elif self is DistanceType.cosine:
            return "cosine"
        elif self is DistanceType.correlation:
            return "correlation"
        else:
            raise ValueError()

    @classmethod
    def from_name(cls, name: str) -> 'DistanceType':
        """Get a distance type from a name."""
        name = name.lower()
        if name == "euclidean":
            return cls.Euclidean
        elif name == "cosine":
            return cls.cosine
        elif name == "correlation":
            return cls.correlation
        else:
            raise NotImplementedError(name)


def distance(u: numpy.ndarray, v: numpy.ndarray, distance_type: DistanceType) -> float:
    """
    Distance from vector u to vector v using the specified distance type.
    """

    if distance_type is DistanceType.Euclidean:
        return _euclidean_distance(u, v)
    elif distance_type is DistanceType.cosine:
        return _cosine_distance(u, v)
    elif distance_type is DistanceType.correlation:
        return _correlation_distance(u, v)
    else:
        raise ValueError()


def _euclidean_distance(u: numpy.ndarray, v: numpy.ndarray):
    """
    Euclidean distance.
    :param u:
    :param v:
    :return:
    """
    return numpy.linalg.norm(u - v)


def _cosine_distance(u: numpy.ndarray, v: numpy.ndarray):
    """
    Cosine distance.
    :param u:
    :param v:
    :return:
    """
    return spatial.distance.cosine(u, v)


def _correlation_distance(u: numpy.ndarray, v: numpy.ndarray):
    """
    Correlation distance.
    :param u:
    :param v:
    :return:
    """
    r = numpy.corrcoef(u, v)[0, 1]
    return 1 - r


def sparse_max(a, b):
    """
    Element-wise maximum for same-sized sparse matrices.
    Thanks to https://stackoverflow.com/a/19318259/2883198
    """

    # Where are elements of b bigger than corresponding element of a?
    b_is_bigger = a - b
    # Pycharm gets type inference wrong here, I'm pretty sure
    # noinspection PyTypeChecker
    b_is_bigger.data = numpy.where(b_is_bigger.data < 0, 1, 0)

    # Return elements of a where a was bigger, and elements of b where b was bigger
    return a - a.multiply(b_is_bigger) + b.multiply(b_is_bigger)


def levenshtein_distance(string_1: str, string_2: str) -> float:
    """
    Levenshtein edit distance between two strings.
    """
    return nltk.edit_distance(string_1, string_2,
                              substitution_cost=1,
                              transpositions=False)


def magnitude_of_negative(c):
    """
    Returns the absolute value of input `c` when it is negative, and 0 otherwise.
    """
    # If c negative
    if c < 0:
        # Make it positive
        return abs(c)
    else:
        # Clamp at zero
        return 0


def binomial_bayes_factor_one_sided(n, k, p0, alternative_hypothesis=">", a=1, b=1):
    """
    Computes one-sided BF for H1: p≠p0 vs H0: p=p0
    Port of https://github.com/jasp-stats/jasp-desktop/blob/development/JASP-Engine/JASP/R/binomialtestbayesian.R#L508
    :param n: trials
    :param k: successes
    :param p0: probability of success under H0
    :param alternative_hypothesis: ">" means H1 is that p>p0, "<" means H1 is p<p0.  Default ">".
    :param a: first parameter of beta-distribution prior on p: B(a,b).  Default 1.
    :param b: second parameter of beta distribution.  Default 1.
    :return: BF_10
    """

    assert alternative_hypothesis in [">", "<"]

    if p0 == 0 and k == 0:
        # In this case k*log(p0) should be 0, but log(0) may cause issues, so we omit it
        log_m_likelihood_h0 = (n - k) * log(1 - p0)
    elif p0 == 1 and k == n:
        # In this case (n - k) * log(1 - p0) should be 0, but log(1 - p0) may cause issues, so we omit it
        log_m_likelihood_h0 = k * log(p0)
    else:
        log_m_likelihood_h0 = k * log(p0) + (n - k) * log(1 - p0)

    if alternative_hypothesis == ">":
        term_1 = log(1 - beta_distribution(a + k, b + n - k).cdf(p0)) + log(beta_function(a + k, b + n - k))
        term_2 = log(beta_function(a, b)) + log(1 - beta_distribution(a, b).cdf(p0))
    elif alternative_hypothesis == "<":
        term_1 = log(beta_distribution(a + k, b + n - k).cdf(p0)) + log(beta_function(a + k, b + n - k))
        term_2 = log(beta_function(a, b)) + log(beta_distribution(a, b).cdf(p0))
    else:
        raise ValueError()

    log_m_likelihood_h1 = term_1 - term_2

    b10 = numpy.exp(log_m_likelihood_h1 - log_m_likelihood_h0)

    return b10


def binomial_bayes_factor_two_sided(n, k, p0, a=1, b=1):
    """
    Computes two-sided BF for H1: p≠p0 vs H0: p=p0
    Port of https://github.com/jasp-stats/jasp-desktop/blob/development/JASP-Engine/JASP/R/binomialtestbayesian.R#L483
    :param n: trials
    :param k: successes
    :param p0: probability of success under H0
    :param a: first parameter of beta-distribution prior on p: B(a,b).  Default 1.
    :param b: second parameter of beta distribution.  Default 1.
    :return: BF_10
    """

    if p0 == 0 and k == 0:
        # In this case k*log(p0) should be 0, but log(0) may cause issues, so we omit it
        log_b10 = log(beta_function(k + a, n - k + b)) - log(beta_function(a, b)) - (n - k) * log(1 - p0)
    elif p0 == 1 and k == n:
        # In this case (n - k) * log(1 - p0) should be 0, but log(1 - p0) may cause issues, so we omit it
        log_b10 = log(beta_function(k + a, n - k + b)) - log(beta_function(a, b)) - k * log(p0)
    else:
        log_b10 = log(beta_function(k + a, n - k + b)) - log(beta_function(a, b)) - k * log(p0) - (n - k) * log(1 - p0)

    b10 = numpy.exp(log_b10)

    return b10


def binomial_bayes_factor(n, k, p0, alternative_hypothesis="≠", a=1, b=1):
    """
    Computes one-sided BF for H1: p≠p0 vs H0: p=p0
    Port of https://github.com/jasp-stats/jasp-desktop/blob/development/JASP-Engine/JASP/R/binomialtestbayesian.R#L546
    :param n: trials
    :param k: successes
    :param p0: probability of success under H0
    :param alternative_hypothesis: Default "≠".
    :param a: first parameter of beta-distribution prior on p: B(a,b).  Default 1.
    :param b: second parameter of beta distribution.  Default 1.
    :return: BF_10
    """

    two_sided_hypotheses         = ["two-sided", "≠", "!=", "=/=", "not-equal", "neq"]
    one_sided_hypotheses_greater = ["greater", "greater-than", "gt", ">"]
    one_sided_hypotheses_lesser  = ["lesser", "less", "less-than", "lt", "<"]

    assert (alternative_hypothesis in two_sided_hypotheses) or (alternative_hypothesis in one_sided_hypotheses_greater) or (alternative_hypothesis in one_sided_hypotheses_lesser)

    if alternative_hypothesis in two_sided_hypotheses:
        return binomial_bayes_factor_two_sided(n, k, p0, a, b)
    elif alternative_hypothesis in one_sided_hypotheses_greater:
        return binomial_bayes_factor_one_sided(n, k, p0, ">", a, b)
    elif alternative_hypothesis in one_sided_hypotheses_lesser:
        return binomial_bayes_factor_one_sided(n, k, p0, "<", a, b)
    else:
        raise ValueError()


def clamp01(x):
    """
    Bounds a value between 0 and 1.
    """
    return max(0, min(1, x))
