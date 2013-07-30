"""Update the Cholesky decomposition of a matrix.

Author:
    Ilias Bilionis

Date:
    2/1/2013

"""

import numpy as np
import scipy.linalg


def update_cholesky(L, B, C):
    """Updates the Cholesky decomposition of a matrix.

    We assume that L is the lower Cholesky decomposition of a matrix A, and
    we want to calculate the Cholesky decomposition of the matrix:
        A   B
        B.T C.
    It can be easily shown that the new decomposition is given by:
        L   0
        D21 D22,
    where
        L * D21.T = B,
    and
        D22 * D22.T = C - D21 * D21.T.

    Arguments:
        L       ---         The Cholesky decomposition of the original
                            n x n matrix.
        B       ---         The n x m upper right part of the new matrix.
        C       ---         The m x m bottom diagonal part of the new matrix.

    Return:
        The lower Cholesky decomposition of the new matrix.
    """
    assert isinstance(L, np.ndarray)
    assert L.ndim == 2
    assert L.shape[0] == L.shape[1]
    assert isinstance(B, np.ndarray)
    assert B.ndim == 2
    assert B.shape[0] == L.shape[0]
    assert isinstance(C, np.ndarray)
    assert C.ndim == 2
    assert B.shape[1] == C.shape[0]
    assert C.shape[0] == C.shape[1]
    n = L.shape[0]
    m = B.shape[1]
    L_new = np.zeros((n + m, n + m))
    L_new[:n, :n] = L
    D21 = L_new[n:, :n]
    D22 = L_new[n:, n:]
    D21[:] = scipy.linalg.solve_triangular(L, B, lower=True).T
    D22[:] = scipy.linalg.cholesky(C - np.dot(D21, D21.T), lower=True)
    return L_new