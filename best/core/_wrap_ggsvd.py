"""Generalized SVD (LAPACK wrappers).

Author:
    Ilias Bilionis

Date:
    8/16/2013
"""


__all__ = ['ggsvd', 'sggsvd', 'dggsvd']


import numpy as np
from ._ggsvd import *


def ggsvd(jobu, jobv, jobq, kl, A, B, alpha, beta, U, V, Q,
          work, iwork):
    if A.dtype == 'float64':
        return dggsvd(jobu, jobv, jobq, kl, A, B, alpha, beta, U, V, Q,
                      work, iwork)
    elif A.dtype == 'float32':
        return sggsvd(jobu, jobv, jobq, kl, A, B, alpha, beta, U, V, Q,
                      work, iwork)
    else:
        raise TypeError()