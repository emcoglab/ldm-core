import logging
import os
import sys

from ..core.corpus import CorpusMetaData

logger = logging.getLogger()


def main():

    # Need to make sure that we over-pad the numbering of the parts so that alphabetical order is numerical order
    target_filename_pattern = "part_{0:06d}.txt"
    lines_per_part = 10_000

    corpus_meta = dict(
        source=CorpusMetaData(
            name="UKWAC",
            path="/Users/cai/Dox/Academic/Analyses/Corpus analysis/UKWAC/1 Text only/cleaned_pre.pos.corpus"),
        target=CorpusMetaData(
            name="UKWAC",
            path="/Users/cai/Dox/Academic/Analyses/Corpus analysis/UKWAC/2 Partitioned"))

    logger.info(f"Loading {corpus_meta['source'].name} corpus from {corpus_meta['source'].path}")

    with open(corpus_meta['source'].path, mode="r", encoding="utf-8") as source_file:

        # initialise some variables
        part_number = 0
        lines = []
        total_line_count = 0

        while True:
            line = source_file.readline()
            if not line:
                break

            lines.append(line)

            if len(lines) >= lines_per_part:
                part_number += 1
                target_path = os.path.join(corpus_meta['target'].path, target_filename_pattern.format(part_number))

                total_line_count += len(lines)

                logger.info(f"Writing next {len(lines):,} lines to {os.path.basename(target_path)}."
                            f" ({total_line_count:,} lines total.)")

                with open(target_path, mode="w", encoding="utf-8") as target_file:
                    target_file.writelines(lines)

                # empty lines
                lines = []


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(module)s | %(message)s', datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    logger.info("Running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")