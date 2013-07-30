"""Update the QR factorization when a row is appended.

Author:
    Ilias Bilionis

Date:
    2/1/2013

"""

import numpy as np


def update_qr(Q, R, W):
    """Update the QR factorization.

    Assume that we have an n x m matrix A such that:
        A = Q * R.
    Consider the (n + 1) x m matrix B such that:
        B = A
            W
    This routine calculates and returns the QR factorization of B.

    Arguments:
        Q       ---     The orthogonal part of the QR factorization of A
                        (n x n) matrix.
        R       ---     The upper triangular part of the QR factorization of A
                        (n x m) matrix.
        W       ---     The matrix that is going to be appended at the bootom
                        of A (k x m) matrix.

    Return:
        Q_new, R_new so that B = Q_new * R_new.

    TODO: Make sure you reimplement this by actually continuing the QR
    factorization of H. Right now it is not very efficient. See page 610 of
    Golub and Van Loan.
    """
    assert isinstance(Q, np.ndarray)
    assert Q.ndim == 2
    assert Q.shape[0] == Q.shape[1]
    assert isinstance(R, np.ndarray)
    assert R.ndim == 2
    assert Q.shape[1] == R.shape[0]
    assert isinstance(W, np.ndarray)
    if W.ndim == 1:
        W = W.reshape((1, W.shape[0]))
    n = Q.shape[0]
    k = W.shape[0]
    H = np.vstack([R, W])
    J, R1 = np.linalg.qr(H, mode='full')
    Z = np.eye(n + k)
    Z[:n, :n] = Q
    Q1 = np.dot(Z, J)
    return Q1, R1
