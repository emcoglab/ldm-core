import logging
import os
import sys

import scipy.io as sio
import scipy.sparse as sps

from .preferences import Preferences
from ..core.corpus.corpus import CorpusMetadata, WindowedCorpus
from ..core.utils.indexing import TokenIndexDictionary

logger = logging.getLogger()


def main():

    out_dir = "/Users/caiwingfield/vectors/n-gram"

    for meta in Preferences.source_corpus_metas:

        for radius in range(1, max(Preferences.window_radii) + 1):

            filename_l = os.path.join(out_dir, f"{meta.name}_r={radius}_left.cooccur")
            filename_r = os.path.join(out_dir, f"{meta.name}_r={radius}_right.cooccur")

            # Skip ones which are already done
            if os.path.isfile(filename_r):
                logger.info(f"Skipping {meta.name} corpus, radius {radius}")
            else:
                logger.info(f"Working on {meta.name} corpus, radius {radius}")

                token_indices = TokenIndexDictionary.load(meta.index_path)

                vocab_size = len(token_indices)

                # Initialise cooccurrence matrices

                # We will store left- and right-cooccurrences separately.
                # At this stage we define radius-n cooccurrence to be words which are /exactly/ n words apart, rather
                # than /up-to/ n words apart.
                # This will greatly speed up computation, and we can sum the values later much faster to get the
                # standard "summed" n-gram counts.

                # First coordinate points to target word
                # Second coordinate points to context word
                cooccur_l = sps.lil_matrix((vocab_size, vocab_size))
                cooccur_r = sps.lil_matrix((vocab_size, vocab_size))

                # Start scanning the corpus
                window_count = 0
                for window in WindowedCorpus(meta, radius):

                    # The target token is the one in the middle, whose index is the radius of the window
                    target_token = window[radius]
                    target_id = token_indices.token2id[target_token]

                    # Count lh occurrences
                    context_token = window[0]
                    context_id = token_indices.token2id[context_token]
                    cooccur_l[target_id, context_id] += 1

                    # Count rh occurrences
                    context_token = window[-1]
                    context_id = token_indices.token2id[context_token]
                    cooccur_r[target_id, context_id] += 1

                    window_count += 1
                    if window_count % 1_000_000 == 0:
                        logger.info(f"\t{window_count:,} tokens processed")

                logger.info("Saving co-occurrence matrices")
                sio.mmwrite(filename_l, cooccur_l)
                sio.mmwrite(filename_r, cooccur_r)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(module)s | %(message)s', datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    logger.info("running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")