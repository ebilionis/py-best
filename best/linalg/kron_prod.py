"""Matrix multiplication with Kronekcer products.

Author:
    Ilias Bilionis

"""


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