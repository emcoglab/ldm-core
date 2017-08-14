"""
===========================
Remove XML markup from BNC XML-formatted documents,
leaving only the linguistic content.
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

import os
import sys
import glob
import logging

from lxml import etree

from ..preferences.preferences import Preferences

logger = logging.getLogger()


def main():
    docs_parent_dir = Preferences.bnc_processing_metas["raw"].path
    out_dir         = Preferences.bbc_processing_metas["detagged"].path

    xsl_filename = os.path.join(os.path.dirname(__file__), "justTheWords.xsl")  # This file is under source control
    xslt = etree.parse(xsl_filename)
    xslt_transform = etree.XSLT(xslt)

    logger.info("Detagging and sorting corpus documents")

    for i, xml_doc_path in enumerate(glob.glob(os.path.join(docs_parent_dir, "**/*.xml"), recursive=True)):

        xml_doc = etree.parse(xml_doc_path)

        new_doc = xslt_transform(xml_doc)

        txt_doc_path = os.path.join(out_dir, os.path.splitext(os.path.basename(xml_doc_path))[0] + ".txt")

        with open(txt_doc_path, mode="w", encoding="utf-8") as target_file:
            target_file.write(str(new_doc))

        if i % 100 == 0:
            logger.info(f"\t{i} files")


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(module)s | %(message)s', datefmt="%Y-%m-%d %H:%M:%S",
                        level=logging.INFO)
    logger.info("running %s" % " ".join(sys.argv))
    main()
    logger.info("Done!")
