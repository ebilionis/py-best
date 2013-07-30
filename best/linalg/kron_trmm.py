"""Multiply a Kronecker product with a matrix.

Author:
    Ilias Bilionis

Date:
    11/26/2012
"""

import numpy as np
import uq.core

def kron_trmm(A, x, uplo=None, trans=None):
    """Solve a triangular system in place.

    A is assumed to be a tuple of numpy arrays.
    x has to be a Fortran array.
    """
    if len(x.shape) == 1:
        x = x.reshape((x.shape[0], 1), order='F')
    if not isinstance(A, tuple):
        aA = ()
        for a in A:
            aA += (a,)
        A = aA
    if uplo is None:
        uplo = 'L' * len(A)
    if trans is None:
        trans = 'N' * len(A)

    uq.core.kron_trmm(uplo, trans, A, x)
