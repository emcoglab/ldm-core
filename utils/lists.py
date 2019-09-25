"""
===========================
Working with lists.
===========================

Dr. Cai Wingfield
---------------------------
Embodied Cognition Lab
Department of Psychology
University of Lancaster
c.wingfield@lancaster.ac.uk
caiwingfield.net
---------------------------
2019
---------------------------
"""


def chunks(l, n):
    """
    Yield successive n-sized chunks from l.
    Thanks, https://stackoverflow.com/a/312464/2883198.
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]
