class CorpusMetadata:
    """
    Corpus metadata
    """

    def __init__(self, name, path, info_path=None):
        self.name = name
        self.path = path
        self.info_path = info_path


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

        self._streamed_corpus = StreamedCorpus(metadata)

    # TODO: does using disjoint "sentences" here lead to unpleasant edge effects?
    def __iter__(self):
        batch = []
        for token in self._streamed_corpus:
            batch.append(token)
            if len(batch) >= self.batch_size:
                yield batch
                # Empty the batch to be refilled
                batch = []
        # Don't forget the last batch, if there is one
        if len(batch) > 0:
            yield batch
