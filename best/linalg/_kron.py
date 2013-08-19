"""Manipulation of linear systems with Kronecker products.

Author:
    Ilias Bilionis

Date:
    19/8/2013
"""


__all__ = ['kron_prod', 'kron_solve']


import numpy as np


def kron_prod(A, x, x_is_1d=False):
    """Matrix multiplication with Kronecker products.

    Note:
        The formula we use is recursive and it is not necessarily the
        most memory efficient way to do this product for non-square
        matrices. However, it is very simple.

    Arguments:
        A   ---    A single matrix or a collection of matrices representing
                   Kronecker product.
        x   ---    A 1D or 2D numpy array to multiply with A.

    Keyword Arguments:
        x_is_1d     ---     This is for internal use. Do not play with it
                            because it will mess up the shape of the output.

    Return:
        The product of the Kronecker product matrix represented by A and
        the numpy array x. Notice that if x is 1D array, then y will be
        a 2D numpy array (column matrix).
    """
    m = A[-1].shape[0]
    n = A[-1].shape[1]
    if len(x.shape) == 1:
        x = x.reshape((x.shape[0], 1), order='F')
    q = x.shape[1]
    X = x.reshape((n, x.shape[0] / n * q), order='F')
    Z = np.dot(A[-1], X) # m x (x.shape[0] / n * q)
    if len(A) == 1:
        return Z
    else:
        s = np.vsplit(Z.T, q)
        st = np.hstack(s)
        r = kron_prod(A[:-1], st, x_is_1d=x_is_1d)
        s = np.vsplit(r.T, q)
        j = np.hstack(s)
        y = j.reshape((np.prod(j.shape) / q, q), order='F')
        return y


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