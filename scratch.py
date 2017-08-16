from .core.corpus.corpus import StreamedCorpus
from .core.corpus.distribution import FreqDist
from .core.model.count import *
from .core.utils.constants import Chirality
from .core.utils.indexing import TokenIndexDictionary
from .preferences.preferences import Preferences


def main():
    toy_corpus = StreamedCorpus(Preferences.source_corpus_metas[-1])
    toy_index = TokenIndexDictionary.load(toy_corpus.metadata.index_path)
    toy_fdist = FreqDist.load(toy_corpus.metadata.freq_dist_path)
    model = PPMIModel(toy_corpus.metadata, Preferences.model_dir, 1, toy_index, toy_fdist)
    model.train(force_retrain=True)

    return


if __name__ == '__main__':
    main()
