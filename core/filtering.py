import re

import nltk


def filter_punctuation(unfiltered_corpus):
    """
    Filters a corpus by ignoring certain punctuation.

    :param unfiltered_corpus:
        A list of tokens, for example that provided by:
            nltk.corpus.PlaintextCorpusReader(unfiltered_corpus_dir, ".+\..+").raw()
    :return:
    """
    return [token
            for token in unfiltered_corpus
            if not re.fullmatch('[' + ignorable_punctuation + ']+', token)]


def filter_frequency(unfiltered_corpus, ignore_tokens_with_frequencies_at_most=0, freq_dist=None):
    """
    Filters a corpus by ignoring words which are too rare.

    :param unfiltered_corpus:
        A list of tokens, for example that provided by:
            nltk.corpus.PlaintextCorpusReader(unfiltered_corpus_dir, ".+\..+").raw()
    :param ignore_tokens_with_frequencies_at_most:
        Ignore any tokens which appear this many times or fewer.
        Set to 0 to include all tokens.
    :param freq_dist:
        Optionally supply an existing frequency distribution to avoid re-computing it
    :return:
    """
    if ignore_tokens_with_frequencies_at_most is 0:
        return unfiltered_corpus
    else:
        if freq_dist is None:
            freq_dist = nltk.probability.FreqDist(unfiltered_corpus)
        return [token
                for token in unfiltered_corpus
                if freq_dist[token] > ignore_tokens_with_frequencies_at_most]

# string.punctuation  = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
# same as above except:
#  - we don't want to ignore:
#       $
#       %
#       &
#       @
#       +
#       =
#  - we do want to ignore
#       … ellipsis
#       – en-dash
#       — em-dash
#       ‘ open single quote
#       ’ close single quote
ignorable_punctuation = r"""!"#'()*,-./:;<>?[\]^_`{|}~…–—‘’"""
