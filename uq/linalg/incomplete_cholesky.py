"""Incomplete Cholesky decomposition.

Author:
    Ilias Bilionis

Date:
    1/30/2013

"""

#from uq.core import pstrf
#from uq.core import zero_tri_part
import numpy as np


def incomplete_cholesky(A, lower=True, in_place=False, tol=-1.):
    """Perform an incomplete Cholesky decomposition of A.

    A is a assumed to be an n x n symmetric semi-positive definite matrix.
    The lower incomplete Cholesky decomposition is defined to be a n x n
    lower triangular matrix L such that:
        P.T * A * P = L * L.T,
    and the upper incomplete Cholesky decomposition is defined to be a n x n
    upper triangular matrix U such that:
        P.T * A * P = U.T * U,
    where P is a permutation matrix and k is the numerical rank of A with a
    prespecified tolerance.

    The routine is basically an interface for lapacks ?pstrf. See _dpstrf for
    the C wrapper of the fortran routine. You must compile that file as a
    library and link it against any LAPACK/BLAS implementation you wish to
    use.

    Arguments:
        A   ---     The matrix to be decomposed.

    Keyword Arguments:
        lower       ---     Look only at the lower triangular part of the
                            matrix.
        in_place    ---     If True then we do the decomposition in place
                            and we do not return a copy.
        tol         ---     The desired tolerance (float). If a negative
                            tolerance is specified, then n * U * max(a(k, k))
                            will be used.

    Caution:
        To guarantee maximum performance make sure that you suply a numpy
        array with Fortran order and type float64. Otherwise, a copy will be
        made anyway.

    Return:
        If in_place is False (default), then the function returns:
            L, P, k    ---     If lower is True.
            U, P, k    ---     If lower is False.
        If in_place is True, then the function:
            Writes L on A if lower is True.
            Writes U on A if lower is False.
        and returns:
            P, k
    """
    assert isinstance(A, np.ndarray)
    assert A.ndim == 2
    assert A.shape[0] == A.shape[1]
    assert isinstance(lower, bool)
    assert isinstance(tol, float)
    A_copy = A
    is_fortran = A.flags['F_CONTIGUOUS'] and A.dtype == 'float64'
    copy_back = in_place and not is_fortran
    if not in_place or not is_fortran:
        A_copy = np.ndarray(A.shape, order='F', dtype='float64')
        A_copy[:] = A
    pinv = np.zeros(A.shape[0], order='F', dtype='int32')
    uplo = 'L' if lower else 'U'
    k = np.ndarray(1, order='F', dtype='int32')
    info = pstrf(uplo, A_copy, pinv, tol, k)
    k = int(k[0])
    P = np.zeros(A.shape, order='F', dtype='int32')
    P[pinv, range(A.shape[0])] = 1
    zero_tri_part(uplo, A_copy, A.shape[0])
    if copy_back:
        A[:, :] = A_copy[:, :]
    if in_place:
        return P, k
    else:
        return A_copy[:, :k] if lower else A_copy[:k, :], P, k
