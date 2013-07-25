"""Calculate the product of a Kronecker matrix with another matrix.

Author:
    Ilias Bilionis

Date:
    11/20/2012

"""

import numpy as np

def kron_prod(A, x):
    """Multiply A with x.

    A is a tuple of matrices representing a Kronecker product.
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
        r = kron_prod(A[:-1], st)
        s = np.vsplit(r.T, q)
        j = np.hstack(s)
        y = j.reshape((np.prod(j.shape) / q, q), order='F')
        return y
