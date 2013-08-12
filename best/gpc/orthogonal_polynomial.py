"""
Describes an orthogonal polynomial.

Author:
    Ilias Bilionis

Date:
    7/25/2013
"""


import numpy as np
import math
import itertools
import best.maps
from quadrature_rule import *
from lancz import *


class OrthogonalPolynomial(best.maps.Function):

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

    def __init__(self, degree, rv=None, left=-1, right=1, wf=lambda(x): 1.,
                 ncap=50, quad=None,
                 name='Orthogonal Polynomial'):
        """Construct the polynomial.

        Keyword Arguments:
            rv      ---     If not None, then it is assumed to be a
                            RandomVariable object which used to define
                            the support interval and the pdf.
            degree  ---     The degree of the polynomial.
            left    ---     The left end of the interval.
            right   ---     The right end of the interval.
            wf      ---     The weight function. The default is the identity.
            ncap    ---     The number of quadrature points.
            quad    ---     A quadrature rule you might want to use in
                            case you are not satisfied with the default
                            one.
            name    ---     A name for the polynomial.
        """
        if rv is not None:
            left, right = rv.interval(1)
            wf = rv.pdf
        if quad is None:
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


class ProductBasis(best.maps.Function):

    """A multi-input orthogonal polynomial basis."""

    # A container of polynomials
    _polynomials = None

    # The total order of the basis
    _degree = None

    # An array of basis terms
    _terms = None

    # The number of terms up to each order
    _num_terms = None

    @property
    def polynomials(self):
        return self._polynomials

    @property
    def degree(self):
        return self._degree

    @property
    def terms(self):
        return self._terms

    @property
    def num_terms(self):
        return self._num_terms

    def __init__(self, rv=None, degree=1, polynomials=None, ncap=50,
                 quad=None, name='Product basis'):
        """Initialize the object.

        Keyword Argument
            rv              ---     If not None, then it is assumed to
                                    be a RandomVectorIndependent.
            degree          ---     The total degree of the basis. Each
                                    one of the polynomials will have this
                                    degree.
            polynomials     ---     We only look at this if rv is None.
                                    A collection of 1D orthogonal
                                    polynomials.
            ncap    ---     The number of quadrature points.
            quad    ---     A quadrature rule you might want to use in
                            case you are not satisfied with the default
                            one.
            name            ---     A name for the basis.
        """
        assert isinstance(degree, int)
        assert degree >= 0
        if rv is not None:
            assert isinstance(rv, best.random.RandomVectorIndependent)
            polynomials = [OrthogonalPolynomial(degree, rv=r, ncap=ncap,
                                                quad=quad) for r in rv.component]
        assert (isinstance(polynomials, tuple) or
                isinstance(polynomials, list))
        for p in polynomials:
            assert isinstance(p, OrthogonalPolynomial)
        self._polynomials = polynomials
        # Find the total order of the basis
        self._degree = max([p.degree for p in polynomials])
        print 'degree: ', self.degree
        # The number of inputs
        num_input = len(polynomials)
        # Compute the basis terms
        num_basis = self._compute_basis_terms()
        super(ProductBasis, self).__init__(num_input, num_basis, name=name)

    def _compute_basis_terms(self):
        """Compute the basis terms.

        The following is taken from Stokhos.

        The approach here for ordering the terms is inductive on the total
        order p.  We get the terms of total order p from the terms of total
        order p-1 by incrementing the orders of the first dimension by 1.
        We then increment the orders of the second dimension by 1 for all of the
        terms whose first dimension order is 0.  We then repeat for the third
        dimension whose first and second dimension orders are 0, and so on.
        How this is done is most easily illustrated by an example of dimension 3:

        Order  terms   cnt  Order  terms   cnt
        0    0 0 0          4    4 0 0  15 5 1
                                 3 1 0
        1    1 0 0  3 2 1        3 0 1
             0 1 0               2 2 0
             0 0 1               2 1 1
                                 2 0 2
        2    2 0 0  6 3 1        1 3 0
             1 1 0               1 2 1
             1 0 1               1 1 2
             0 2 0               1 0 3
             0 1 1               0 4 0
             0 0 2               0 3 1
                                 0 2 2
        3    3 0 0  10 4 1       0 1 3
             2 1 0               0 0 4
             2 0 1
             1 2 0
             1 1 1
             1 0 2
             0 3 0
             0 2 1
             0 1 2
             0 0 3
        """
        # Number of inputs
        num_dim = len(self.polynomials)

        # Temporary array of terms grouped in terms of same order
        terms_order = [[] for i in range(self.degree + 1)]

        # Store number of terms up to each order
        self._num_terms = np.zeros(self.degree + 2, dtype='i')

        # Set order zero
        terms_order[0] = ([np.zeros(num_dim, dtype='i')])
        self.num_terms[0] = 1

        # The array cnt stores the number of terms we need to
        # increment for each dimension.
        cnt = np.zeros(num_dim, dtype='i')
        for j, p in itertools.izip(range(num_dim), self.polynomials):
            if p.degree >= 1:
                cnt[j] = 1

        cnt_next = np.zeros(num_dim, dtype='i')
        term = np.zeros(num_dim, dtype='i')

        # Number of basis functions
        num_basis = 1

        # Loop over orders
        for k in range(1, self.degree + 1):
            self.num_terms[k] = self.num_terms[k - 1]
            # Stores the inde of the term we are copying
            prev = 0
            # Loop over dimensions
            for j, p in itertools.izip(range(num_dim), self.polynomials):
                # Increment orders of cnt[j] terms for dimension j
                for i in range(cnt[j]):
                    if terms_order[k - 1][prev + i][j] < p.degree:
                        term = terms_order[k - 1][prev + i].copy()
                        term[j] += 1
                        terms_order[k].append(term)
                        num_basis += 1
                        self.num_terms[k] += 1
                        for l in range(j + 1):
                            cnt_next[l] += 1
                if j < num_dim - 1:
                    prev += cnt[j] - cnt[j + 1]
            cnt[:] = cnt_next
            cnt_next[:] = 0
        self.num_terms[self.degree + 1] = num_basis
        # Copy into final terms array
        self._terms = []
        for k in range(self.degree + 1):
            num_k = len(terms_order[k])
            for j in range(num_k):
                self._terms.append(terms_order[k][j])
        return num_basis

    def __str__(self):
        """Return a string representation of the object."""
        s = super(ProductBasis, self).__str__() + '\n'
        s += 'sz = ' + str(self.num_output) + '\n'
        for i in range(self.num_output):
            s += str(i) + ': '
            for j in range(self.num_input):
                s += str(self.terms[i][j]) + ' '
            s += '\n'
        s += 'num_terms = '
        for i in range(self.degree + 1):
            s += str(self.num_terms[i]) + ' '
        return s

    def __call__(self, x):
        """Evaluate the polynomials at x."""
        x = np.atleast_2d(x)
        phi = np.ndarray((x.shape[0], self.num_output))
        basis_eval_tmp = [[] for j in range(self.num_input)]
        for j in range(self.num_input):
            basis_eval_tmp[j] = self.polynomials[j](x[:, j])
        for k in range(self.num_output):
            phi[:, k] = np.ones(x.shape[0])
            for j in range(self.num_input):
                phi[:, k] *= basis_eval_tmp[j][:, self.terms[k][j]]
        return phi