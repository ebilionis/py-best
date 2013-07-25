"""Define the Separable Covariance Function

Author:
    Ilias Bilionis

Date:
    11/20/2012

"""

import numpy as np
import itertools as iter
from uq.gp import CovarianceFunction


class SeparableCovarianceFunction(CovarianceFunction):

    """A Separable Covariance Function."""

    # A tuple of covariance functions
    _cov = None
    
    @property
    def cov(self):
        """Return covariance."""
        return self._cov

    @cov.setter
    def cov(self, value):
        """Set the covariance functions."""
        if not isinstance(value, tuple):
            raise TypeError('value must be a tuple.')
        for c in value:
            if not isinstance(c, CovarianceFunction):
                raise TypeError(
                    'value is not a tuple of covariance functions.')
        self._cov = value

    def __init__(self, cov=None, name='SeparableCovarianceFunction'):
        """Initialize the object.
        
        Keyword Arguments:
        cov     ---     A tuple of covariance functions

        """
        super(SeparableCovarianceFunction, self).__init__(name=name)
        if cov is not None:
            self.cov = cov

    def _check_if_tuple(self, x, var_name):
        """Check if x is a tuple of the right length"""
        if not isinstance(x, tuple):
            raise TypeError(var_name +
                    ' must be a tuple.')
        if not len(x) == len(self.cov):
            raise TypeError(var_name +
                    ' must have the same number of components as the number'
                    + ' of covariance functions.')

    def __call__(self, hyp, x1, x2=None, A=None):
        """Evaluate the covariance function."""
        self._check_if_tuple(hyp, 'hyp')
        self._check_if_tuple(x1, 'x1')
        if x2 is None:
            x2 = ()
            for _ in xrange(len(self.cov)):
                x2 += (None, )
        else:
            self._check_if_tuple(x2, 'x2')
        if A is None:
            A = ()
            for c, chyp, cx1, cx2 in iter.izip(self.cov, hyp, x1, x2):
                A += (c(chyp, cx1, cx2), )
            return A
        else:
            self._check_if_tuple(A, 'A')
            for c, chyp, cx1, cx2, cA in iter.izip(self.cov, hyp, x1, x2, A):
                c(chyp, cx1, cx2, cA)

    def __str__(self):
        """Return a string representation of the object."""
        s = super(SeparableCovarianceFunction, self).__str__()
        for c in self.cov:
            s += '\n' + str(c)
        return s
