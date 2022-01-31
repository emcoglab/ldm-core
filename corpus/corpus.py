"""
===========================
Classes related to corpora.
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

import glob
import os


class CorpusMetadata:
    """Corpus metadata."""
    def __init__(self, name, path, freq_dist_path=None):
        # The name of the corpus
        self.name = name
        # The path to the corpus
        self.path = path
        # The path to the frequency distribution for the corpus,
        # used to canonically index the corpus's unique tokens
        self.freq_dist_path = freq_dist_path


class StreamedCorpus:
    """Corpus which yields individual tokens."""

    def __init__(self, path: str):
        """
        :param path: Must point to a tokenised corpus: a file or set of files containing one token on each line.
        :param metadata:
        """
        self.path: str = path

    def __iter__(self):
        # If it's a single file, open it
        if os.path.isfile(self.path):
            with open(self.path, mode="r", encoding="utf-8") as corpus_file:
                # We can do this because the corpus is tokenised,
                # so there's a single token on each line
                for token in corpus_file:
                    yield token.strip()
        # If it's a directory of files, open them in sequence
        elif os.path.isdir(self.path):
            # Files which aren't .DS_Store
            file_paths = glob.glob(os.path.join(self.path, "+.*"))
            for file_path in file_paths:
                with open(file_path, mode="r", encoding="utf-8") as corpus_file:
                    for token in corpus_file:
                        yield token.strip()
        else:
            raise FileNotFoundError(f"{self.path} does not exist.")


class WindowedCorpus:
    """Corpus presented through a sliding window."""

    def __init__(self, path: str, window_width: int):
        """
        :param path: Must point to a tokenised corpus.
        :param window_width:
        """
        self.path: str = path
        self.window_width: int = window_width

    def __iter__(self):
        window = []
        token_count = 0
        for token in StreamedCorpus(self.path):

            # Fill up the initial window, such that the next token to be read will produce the first full window
            if token_count < self.window_width - 1:
                window.append(token)
                token_count += 1

            # Each iteration of this loop will advance the position of the window by one
            else:

                # Add a new token on the rhs of the window
                window.append(token)
                # The window is now full

                yield window

                # Pop the lhs token out of the window to await the next one to be added, which
                # will cause the window to have moved exactly one token over in the corpus
                del window[0]


class BatchedCorpus:
    """Corpus which yields batches of tokens."""

    def __init__(self, path: str, batch_size: int):
        """
        :param path: Must point to a tokenised corpus.
        :param batch_size: Size of batch
        """
        self.path: str = path
        self.batch_size: int = batch_size

    def __iter__(self):
        batch = []
        for token in StreamedCorpus(self.path):
            batch.append(token)
            if len(batch) >= self.batch_size:
                yield batch
                # Empty the batch to be refilled
                batch = []
        # Don't forget the last batch, if there is one
        if len(batch) > 0:
            yield batch
