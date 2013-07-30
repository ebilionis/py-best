"""Solve a triangular system of Kronecker products in place.

Author:
    Ilias Bilionis

Date:
    11/20/2012

"""


import numpy as np


def kron_solve(A, y):
    """Solve a triangular system.

    The system we are attempting to solve is:
        kron(A[0], A[1], ...) * x = y.

    Note:
        Ideally we would like to see if the matrices are of a particular
        type (symmetric, upper/lower triangular, etc.). This is much simpler
        though.

    Arguments:
        A   ---     A single 2D numpy array or a collection of them
                    representing a Kronecker product.
        y   ---     The right hand side of the linear system.

    Return:
        The solution of the linear system.
    """
    n = A[-1].shape[0]
    if len(y.shape) == 1:
        y = y.reshape((y.shape[0], 1), order='F')
    Y = y.reshape((n, y.shape[0] / n * y.shape[1]), order='F')
    Z = np.linalg.solve(A[-1], Y)
    if len(A) == 1:
        return Z.reshape(y.shape, order='F')
    else:
        X = kron_solve(A[:-1], Z.T.copy()).T
        return X.reshape(y.shape, order='F')