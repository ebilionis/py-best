"""Solve a triangular system of Kronecker products in place.

Author:
    Ilias Bilionis

Date:
    11/20/2012

"""

import numpy as np
def kron_solve(A, x):
    """Solve a triangular system.

    A is assumed to be a tuple of numpy arrays.
    """
    n = A[-1].shape[0]
    if len(x.shape) == 1:
        x = x.reshape((x.shape[0], 1), order='F')
    X = x.reshape((n, x.shape[0] / n * x.shape[1]), order='F')
    Z = np.linalg.solve(A[-1], X)
    if len(A) == 1:
        return Z.reshape(x.shape, order='F')
    else:
        Y = kron_solve(A[:-1], Z.T.copy()).T
        return Y.reshape(x.shape, order='F')
