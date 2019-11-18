"""
===========================
Working with lists and iterables.
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


def unzip(l):
    """
    Sort-of undoes a zip.

    Example:
        a = [1, 2, 3]
        b = ['a', 'b', 'c']
        z = list(zip(a, b))  # [(1, 'a'), (2, 'b'), (3, 'c')]
        c, d = unzip(z)  # (1, 2, 3), ('a', 'b', 'c')

    Note: Outputs tuples
    Note: Doesn't work with empty lists.

    See: https://stackoverflow.com/questions/19339/transpose-unzip-function-inverse-of-zip
    """
    return zip(*l)
