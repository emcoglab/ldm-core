"""
===========================
File saving and loading.
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

import sys

import numpy
import scipy.sparse


def load_npz_with_mmap(file, memory_map: bool = False):
    """
    Copied from scipy.sparse.load_npz, except it allows memory mapping.
    """
    if memory_map:
        with numpy.load(file=file,
                        mmap_mode="r" if memory_map else None,
                        # Works only with Numpy version >= 1.10.0
                        allow_pickle=False
                        ) as loaded:
            try:
                matrix_format = loaded['format']
            except KeyError:
                raise ValueError('The file {} does not contain a sparse matrix.'.format(file))

            matrix_format = matrix_format.item()

            if sys.version_info[0] >= 3 and not isinstance(matrix_format, str):
                # Play safe with Python 2 vs 3 backward compatibility;
                # files saved with Scipy < 1.0.0 may contain unicode or bytes.
                matrix_format = matrix_format.decode('ascii')

            try:
                cls = getattr(scipy.sparse, '{}_matrix'.format(matrix_format))
            except AttributeError:
                raise ValueError('Unknown matrix format "{}"'.format(matrix_format))

            if matrix_format in ('csc', 'csr', 'bsr'):
                return cls((loaded['data'], loaded['indices'], loaded['indptr']), shape=loaded['shape'])
            elif matrix_format == 'dia':
                return cls((loaded['data'], loaded['offsets']), shape=loaded['shape'])
            elif matrix_format == 'coo':
                return cls((loaded['data'], (loaded['row'], loaded['col'])), shape=loaded['shape'])
            else:
                raise NotImplementedError('Load is not implemented for '
                                          'sparse matrix of format {}.'.format(matrix_format))
    else:
        # Use standard load function
        return scipy.sparse.load_npz(file)
