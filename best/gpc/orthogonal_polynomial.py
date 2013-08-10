"""
Describes an orthogonal polynomial.

Author:
    Ilias Bilionis

Date:
    7/25/2013
"""


import numpy as np
import math
from best.maps import Function
from quadrature_rule import *
from lancz import *


class OrthogonalPolynomial(Function):

    """1D Orthogonal Polynomial via recursive relation.

    A polynomial is of course a function.
    """

    # Recurrence coefficient alpha
    _alpha = None

    # Recurrence coefficient beta
    _beta = None

    # Recurrence coefficient gamma
    _gamma = None

    # Is the polynomial normalized
    _is_normalized = None

    @property
    def degree(self):
        """Return the degree of the polynomial."""
        return self.alpha.shape[0] - 1

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta

    @property
    def gamma(self):
        return self._gamma

    @property
    def is_normalized(self):
        return self._is_normalized

    def __init__(self, degree, left=-1, right=1, wf=lambda(x): 1.,
                 ncap=50, quad=QuadratureRule,
                 name='Orthogonal Polynomial'):
        """Construct the polynomial.

        Keyword Arguments:
            degree  ---     The degree of the polynomial.
            left    ---     The left end of the interval.
            right   ---     The right end of the interval.
            wf      ---     The weight function. The default is the identity.
            ncap    ---     The number of quadrature points.
            name    ---     A name for the polynomial.
        """
        quad = QuadratureRule(left=left, right=right, wf=wf, ncap=ncap)
        self._alpha, self._beta = lancz(quad.x, quad.w, degree + 1)
        self._gamma = np.ones(self.degree + 1)
        self.normalize()
        super(OrthogonalPolynomial, self).__init__(1, self.degree + 1, name=name)

    def __call__(self, x):
        """Evaluate the polynomial basis at x."""
        if not isinstance(x, np.ndarray):
            x = np.array([float(x)])
        phi = np.zeros((x.shape[0], self.degree + 1)) # N x (P + 1)
        phi[:, 0] = 1. / self.gamma[0]
        if self.degree >= 1:
            phi[:, 1] = (x - self.alpha[0]) * (phi[:, 0] / self.gamma[1])
        for i in range(2, self.degree + 1):
            phi[:, i] = ((x - self.alpha[i - 1]) * phi[:, i - 1] -
                self.beta[i - 1] * phi[:, i - 2]) / self.gamma[i]
        return phi

    def d(self, x, return_eval=False):
        """Evaluate the derivative of the polynomial.

        Arguments:
            x   ---     The input point(s).

        Keyword Arguments:
            return_eval --- If set to True, then return also the
                            polynomials themselves.
        """
        if not isinstance(x, np.ndarray):
            x = np.array([float(x)])
        phi = self(x)
        dphi = np.zeros((x.shape[0], self.degree + 1))
        if self.degree >= 1:
            dphi[:, 1] = phi[:, 0] / self.gamma[1]
        for i in range(2, self.degree + 1):
            dphi[:, i] = ((phi[:, i - 1] +
                           (x - self.alpha[i - 1]) * dphi[:, i - 1] -
                           self.beta[i - 1] * dphi[:, i - 2]
                          ) / self.gamma[i])
        if return_eval:
            return phi, dphi
        return dphi

    def _evaluate_square_norms(self):
        """Evaluate the square norms of the polynomials."""
        s_norm = np.zeros(self.degree + 1)
        s_norm[0] = self.beta[0] / (self.gamma[0] ** 2)
        for i in range(1, self.degree + 1):
            s_norm[i] = (self.beta[i] / self.gamma[i]) * s_norm[i - 1]
        return s_norm

    def normalize(self):
        """Normalize the polynomials."""
        self.beta[0] = math.sqrt(self.beta[0])
        self.gamma[0] = self.beta[0]
        for i in range(1, self.degree + 1):
            self.beta[i] = math.sqrt(self.beta[i] * self.gamma[i])
            self.gamma[i] = self.beta[i]
        self._is_normalized = True

    def _to_string(self, pad):
        """Return a string representation of the object."""
        s = super(OrthogonalPolynomial, self)._to_string(pad) + '\n'
        s += pad + ' alpha: ' + str(self.alpha) + '\n'
        s += pad + ' beta: ' + str(self.beta) + '\n'
        s += pad + ' gamma: ' + str(self.gamma) + '\n'
        s += pad + ' normalized: ' + str(self.is_normalized)
        return s