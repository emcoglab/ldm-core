class CorpusMetadata:
    """
    Corpus metadata
    """

    def __init__(self, name, path, info_path=None, index_path=None):
        self.name = name
        self.path = path
        self.info_path = info_path
        self.index_path = index_path


class StreamedCorpus(object):
    """
    Corpus which yields individual tokens
    """

    def __init__(self, metadata: CorpusMetadata):
        """

        :type metadata: CorpusMetadata
        :param metadata:
        """
        self.metadata = metadata

    def __iter__(self):
        with open(self.metadata.path, mode="r", encoding="utf-8") as corpus_file:
            for token in corpus_file:
                yield token.strip()


class WindowedCorpus(object):
    """
    Corpus presented through a sliding window
    """

    def __init__(self, metadata: CorpusMetadata, window_radius: int):
        self.metadata = metadata
        self.window_radius = window_radius

    def __iter__(self):

        # We will load in the full window and count left and right cooccurences separately
        # The size of the symmetric window is twice the radius, plus 1 (centre)
        window_diameter = 2 * self.window_radius + 1

        window = []
        token_count = 0
        for token in StreamedCorpus(self.metadata):

            # Fill up the initial window, such that the next token to be read will produce the first full window
            if token_count < window_diameter:
                window.append(token)
                token_count += 1

            # Each iteration of this loop will advance the position of the window by one
            else:

                # Add a new token on the rhs of the window
                window.append(token)

                # The window is now full
                yield window

                # Pop the lhs token out of the window to await the next one, which will cause the window to have
                # moved exactly one token over in the corpus
                window = window[1:]


class BatchedCorpus(object):
    """
    Corpus which yields batches of tokens
    """

    def __init__(self, metadata: CorpusMetadata, batch_size: int):
        """

        :type batch_size: int
        :type metadata: CorpusMetadata
        :param metadata:
        :param batch_size:
        Size of batch
        """
        self.metadata = metadata
        self.batch_size = batch_size

    def __iter__(self):
        batch = []
        for token in StreamedCorpus(self.metadata):
            batch.append(token)
            if len(batch) >= self.batch_size:
                yield batch
                # Empty the batch to be refilled
                batch = []
        # Don't forget the last batch, if there is one
        if len(batch) > 0:
            yield batch