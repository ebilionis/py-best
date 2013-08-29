"""A wrapper for pstrf.

Author:
    Ilias Bilionis

Date:
    8/28/2013
"""


__all__ = ['pstrf', 'spstrf', 'dpstrf']


import numpy as np
from ._pstrf import *


def pstrf(uplo, A, P, tol, work):
    rank = np.ndarray(1, dtype='int32')
    if A.dtype == 'float32':
        info = spstrf(uplo, A, P, rank, tol, work)
    elif A.dtype == 'float64':
        info = dpstrf(uplo, A, P, rank, tol, work)
    else:
        raise TypeError()
    return rank[0], info