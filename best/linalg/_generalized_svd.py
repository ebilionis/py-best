"""Generalized SVD (LAPACK wrappers).

Author:
    Ilias Bilionis

Date:
    8/16/2013
"""


__all__ = ['GeneralizedSVD']


import numpy as np
from .. import Object
from ..core import lapack


class GeneralizedSVD(Object):

    """A class that represents the generalized svd decomposition of A and B."""

    # Internal A
    _A = None

    # Internal B
    _B = None

    # Internal alpha
    _alpha = None

    # Internal beta
    _beta = None

    # Internal U
    _U = None

    # Internal V
    _V = None

    # Internal Q
    _Q = None

    # Internal work
    _work = None

    # Internal iwork
    _iwork = None

    # Internal k
    _k = None

    # Internal l
    _l = None

    # Internal R
    _R = None

    # Internal C
    _C = None

    # Internal S
    _S = None

    # Internal D1
    _D1 = None

    # Internal D2
    _D2 = None

    @property
    def A(self):
        return self._A

    @property
    def B(self):
        return self._B

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta

    @property
    def U(self):
        return self._U

    @property
    def V(self):
        return self._V

    @property
    def Q(self):
        return self._Q

    @property
    def m(self):
        return self.A.shape[0]

    @property
    def n(self):
        return self.A.shape[1]

    @property
    def p(self):
        return self.B.shape[0]

    @property
    def work(self):
        return self._work

    @property
    def iwork(self):
        return self._iwork

    @property
    def k(self):
        return self._k

    @property
    def l(self):
        return self._l

    @property
    def R(self):
        return self._R

    @property
    def C(self):
        return self._C

    @property
    def S(self):
        return self._S

    @property
    def D1(self):
        return self._D1

    @property
    def D2(self):
        return self._D2

    def __init__(self, A, B, do_U=True, do_V=True, do_Q=True,
                 name='Generalized SVD'):
        """Initialize the object and perform the decomposition.

        Arguments:
            A       ---     Numpy array (a copy will be made).
            B       ---     Numpy array (a copy will be made).

        Keyword Arguments
            do_U    ---     Compute U if True.
            do_V    ---     Compute V if True.
            do_Q    ---     Compute Q if True.
            name    ---     A name for the class.
        """
        super(GeneralizedSVD, self).__init__(name=name)
        if A.dtype == 'float32':
            ggsvd = lapack.wsggsvd
        else:
            ggsvd = lapack.wdggsvd
        if do_U:
            jobU = 'U'
        else:
            jobU = 'N'
        if do_V:
            jobV = 'V'
        else:
            jobV = 'N'
        if do_Q:
            jobQ = 'Q'
        else:
            jobQ = 'N'
        print ggsvd.__doc__
        print A.shape
        print B.shape
        (self._k, self._l, self._A, self._B, self._alpha, self._beta,
         self._U, self._V, self._Q, self._sort, info) = ggsvd(A, B)
        if self.m - self.k - self.l >= 0:
            # Construct R
            self._R = np.array(self.A[:self.k + self.l,
                                 self.n - self.k - self.l:],
                               copy=True)
            self._R[self.k:, :self.k] = 0.
            # Construct C
            self._C = np.diag(self.alpha[self.k:self.k + self.l])
            self._S = np.diag(self.beta[self.k:self.k + self.l])
            self._D1 = np.zeros((self.m, self.k + self.l))
            self.D1[:self.k, :self.k] = np.eye(self.k)
            self.D1[self.k:self.k + self.l, self.k:] = self.C
            self._D2 = np.zeros((self.p, self.k + self.l))
            self.D2[:self.l, self.k:] = self.S
        else:
            self._R = np.ndarray((self.k + self.l, self.k + self.l))
            self.R[:self.m, :] = self.A[:,
                                        self.n - self.k - self.l:]
            self.R[self.k:self.m - self.k, :self.k] = 0.
            self.R[self.m:, self.m:] = self.B[self.m - self.k:self.l,
                                              self.n - self.m - self.k
                                              - self.l:]
            self._C = np.diag(self.alpha[self.k:self.m])
            self._S = np.diag(self.beta[self.k:self.m])
            self._D1 = np.zeros((self.m, self.k + self.l))
            self.D1[:self.k, :self.k] = np.eye(self.k)
            self.D1[self.k:, self.k:self.m] = self.C
            self._D2 = np.zeros((self.p, self.k + self.l))
            self.D2[:self.m - self.k, self.k:self.m] = self.S
            self.D2[self.m - self.k:self.l, self.m:] = np.eye(self.k + self.l - self.m)