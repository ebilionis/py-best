"""Incomplete Cholesky decomposition.

Author:
    Ilias Bilionis

Date:
    1/30/2013

"""


__all__ = ['IncompleteCholesky']


import numpy as np
from .. import Object
from ..core import pstrf


class IncompleteCholesky(Object):

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
    """

    # Internal A
    _A = None

    # Internal piv
    _piv = None

    # Internal P
    _P = None

    # Internal L
    _L = None

    # Internal U
    _U = None

    # rank
    _rank = None

    @property
    def A(self):
        return self._A

    @property
    def piv(self):
        return self._piv

    @property
    def P(self):
        return self._P

    @property
    def L(self):
        return self._L

    @property
    def rank(self):
        return self._rank

    def __init__(self, A, lower=True, tol=-1., name='Incomplete Cholesky'):
        """Initialize the object.

        Arguments:
            A   ---     The matrix whose decomposition you seek. It will
                        be copied internally.

        Keyword Arguments:
            lower   ---     If True, then compute the lower incomplete
                            Cholesky. Otherwise compute the upper
                            incomplete Cholesky.
            tol     ---     The desired tolerance (float). If a negative
                            tolerance is specified, then n * U * max(a(k, k))
                            will be used.
            name    ---     A name for the object.
        """
        super(IncompleteCholesky, self).__init__(name=name)
        if A.dtype == 'float32' or A.dtype == 'float64':
            dtype = A.dtype
        else:
            dtype = 'float64'
        self._A = np.ndarray(A.shape, order='F', dtype=dtype)
        self._A[:] = A
        if lower:
            uplo = 'L'
        else:
            uplo = 'U'
        n = A.shape[0]
        work = np.ndarray(2 * n, dtype=dtype)
        self._piv = np.ndarray(n, dtype='int32')
        self._rank, info = pstrf(uplo, self.A, self.piv, tol, work)
        self._P = np.zeros((n, n), dtype='int32')
        self.P[self.piv, range(n)] = 1
        if lower:
            self._L = np.tril(self.A)[:, :self.rank]
        else:
            self._U = np.triu(self.A)[:self.rank, :]