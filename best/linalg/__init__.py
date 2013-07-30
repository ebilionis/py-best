"""
.. module: best.linalg
    :synopsis: Some linear algebra routines.

.. moduleauthor: Ilias Bilionis <ebilionis@gmail.com>

The linear module defines several routines that cannot be found in numpy or
scipy but are extremely useful in various Bayesian problems.

.. automodule:: best.linalg.kron_prod
   :members: kron_prod

"""

import scipy.linalg
import numpy as np


from kron_prod import *


def kron_prod(A, x):
    """Matrix multiplication with Kronecker products.

    The method computes the product:
        .. math::
            \mathbf{y} = (\otimes_{i=1}^s \mathbf{A}_i) \mathbf{x},
            
    where :math:`\mathbf{A}_i` are suitable matrices.
    The characteristic of the routine is that it does not form the
    Kronecker product explicitly. Also, :math:`\mathbf{x}` can be a
    matrix of appropriate dimensions. Of course, it will throw an
    exception if you don't have the dimensions right.

    :param A: A collection of matrices whose Kronecker product is
			  assumed. If A is not a collection (list or tuple), but
			  a simple matrix, then the method just performs a simple
			  matrix-vector multiplication.
    :param x: A vector or a matrix.
    :returns: A numpy array containing the result of the computation.

    Here is an example:

    >>> A1 = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
    >>> A2 = A1
    >>> A = (A1, A2)
    >>> x = np.ones(A1.shape[1] * A2.shape[1])
    >>> y = best.linalg.kron_prod(A, x)
    """
    if not (isinstance(A, tuple) or isinstance(A, list)):
		A = (A, )
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

def kron_solve(A, y):
    """Solve a linear system involving Kronecker products.

    The methods solves the following linear system:
		.. math::
			(\otimes_{i=1}^s\mathbf{A}_i)\mathbf{x} = \mathbf{y},

	where :math:`\mathbf{A}_i` are suitable matrices and
	:math:`\mathbf{y}` is a vector or a matrix.

    :param A: A collection of matrices whose Kronecker product is
			  assumed. If A is not a collection (list or tuple), but
			  a simple matrix, then the method just solves a simple
			  linear system.
    :param y: A vector or a matrix.
    :returns: A numpy array containing the result of the computation.

    Here is an example:

    >>> A1 = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
    >>> A2 = A1
    >>> A = (A1, A2)
    >>> y = np.random.randn(A1.shape[1] * A2.shape[1])
    >>> x = best.linalg.kron_solve(A, y)
    """
    if not (isinstance(A, tuple) or isinstance(A, list)):
		A = (A, )
    n = A[-1].shape[0]
    if len(y.shape) == 1:
        y = y.reshape((y.shape[0], 1), order='F')
    Y = y.reshape((n, y.shape[0] / n * y.shape[1]), order='F')
    Z = np.linalg.solve(A[-1], Y)
    if len(A) == 1:
        return Z.reshape(y.shape, order='F')
    else:
        X = kron_solve(A[:-1], Z.T.copy()).T
        return X.reshape(x.shape, order='F')


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


def update_cholesky_linear_system(x, L_new, z):
    """Update the solution of Cholesky-solved linear system.

    Assume that originally we had an :math:`n\\times n` lower triangular
    matrix :math:`\mathbf{L}` and that we have already solved the linear
    system:
		.. math::
			\mathbf{L} \mathbf{x} = \mathbf{y},

    Now, we wish to solve the linear system:
		.. math::
			\mathbf{L}'\mathbf{x}' = \mathbf{y}',

    where :math:`\mathbf{L}` is again lower triangular matrix whose
    top :math:`n \\times n` component is identical to :math:`\mathbf{L}`
    and :math:`\mathbf{y}'` is :math:`(\mathbf{y}, \mathbf{z})`. The
    solution is:
		.. math::
			\mathbf{x}' = (\mathbf{x}, \mathbf{x}_u),

    where :math:`\mathbf{x}_u` is the solution of the triangular system:
		.. math::
			\mathbf{L}_{22}' * \mathbf{x}_u = \mathbf{z} - \mathbf{L}_{21}' \mathbf{x},

    where :math:`\mathbf{L}_{22}'` is the lower :math:`m\\times m`
    component of :math:`\mathbf{L}'` and :math:`\mathbf{L}_{21}'` is the
    :math:`m\\times n` bottom left component of :math:`\mathbf{L}'`.

	:param x: The solution of the first Cholesky system.
	:param L_new: The new Cholesky factor.
	:param z: The new part of :math:`\mathbf{y}`.
	:returns: A numpy array containing the result.

    """
    assert isinstance(x, np.ndarray)
    assert x.ndim <= 2
    regularized_x = False
    if x.ndim == 1:
        regularized_x = True
        x = x.reshape((x.shape[0], 1))
    assert isinstance(L_new, np.ndarray)
    assert L_new.shape[0] == L_new.shape[1]
    assert isinstance(z, np.ndarray)
    assert z.ndim <= 2
    regularized_z = False
    if z.ndim == 1:
        regularized_z = True
        z = z.reshape((z.shape[0], 1))
    assert x.shape[1] == z.shape[1]
    assert L_new.shape[0] == x.shape[0] + z.shape[0]
    n = x.shape[0]
    D22 = L_new[n:, n:]
    D21 = L_new[n:, :n]
    x_u = scipy.linalg.solve_triangular(D22, z - np.dot(D21, x), lower=True)
    y = np.vstack([x, x_u])
    if regularized_x or regularized_z:
        y = y.reshape((y.shape[0],))
    return y


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


def incomplete_cholesky(A, lower=True, in_place=False, tol=-1.):
    """Perform an incomplete Cholesky decomposition of A.

    :math:`\mathbf{A}` is a assumed to be an :math:`n\\times n`
    symmetric semi-positive definite matrix.
    The lower incomplete Cholesky decomposition is defined to be a\
    :math:`n\\times n` lower triangular matrix :math:`\mathbf{L}` such
    that:
		.. math::
			\mathbf{P}^T \mathbf{A}\mathbf{P} = \mathbf{L}\mathbf{L}^T,

    and the upper incomplete Cholesky decomposition is defined to be a
    :math:`n \\times n` upper triangular matrix :math:`\mathbf{U}` such
    that:
		.. math::
			\mathbf{P}^T \mathbf{A}\mathbf{P} = = \mathbf{U}^T\mathbf{U}

    where :math:`\mathbf{P}` is a permutation matrix and
    :math:`k` is the numerical rank of :math:`\mathbf{A}` with a
    prespecified tolerance.

    The method is basically an interface for lapacks ?pstrf. See _dpstrf for
    the C wrapper of the fortran routine. You must compile that file as a
    library and link it against any LAPACK/BLAS implementation you wish to
    use.

	:param A: 			The matrix to be decomposed.
	:param lower:		Look only at the lower triangular part of the matrix
						(True or False).
	:param in_place:	If True then we do the decomposition in place
						and we do not return a copy.
	:returns:			If in_place is False (default), then the function returns

    L, P, k    ---     If lower is True.
            U, P, k    ---     If lower is False.
        If in_place is True, then the function:
            Writes L on A if lower is True.
            Writes U on A if lower is False.
        and returns:
            P, k

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
