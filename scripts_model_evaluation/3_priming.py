"""
===========================
Evaluate using priming data.
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

from ..core.evaluation.priming import SppRegressionTester, SppData
from ..core.model.count import LogNgramModel
from ..core.utils.indexing import TokenIndexDictionary
from ..core.utils.logging import log_message, date_format
from ..preferences.preferences import Preferences

logger = logging.getLogger(__name__)


def main():

    spp_data = SppData()

    for corpus_metadata in Preferences.source_corpus_metas:

        token_index = TokenIndexDictionary.load(corpus_metadata.index_path)

        for window_radius in Preferences.window_radii:

            model = LogNgramModel(corpus_metadata, window_radius, token_index)
            model.train()

            logger.info(f"Missing words for {model}:")
            for word in spp_data.missing_words(model):
                logger.info(f"\t{word}")

            spp_data.add_model_predictor(model)


if __name__ == "__main__":
    logging.basicConfig(format=log_message, datefmt=date_format, level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
