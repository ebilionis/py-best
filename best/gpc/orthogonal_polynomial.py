"""
Describes an orthogonal polynomial.

Author:
    Ilias Bilionis
    
Date:
    7/25/2013
"""


import numpy as np


class OrthogonalPolynomial(object):
    # Polynomial order
    _p = None
    
    # Recurrence coefficient alpha
    _alpha = None
    
    # Recurrence coefficient beta
    _beta = None
    
    # Recurrence coefficient delta
    _delta = None
    
    # Is the polynomial normalized
    _is_normalized = None
    
    @property
    def degree(self):
        return self._p
    
    @property
    def alpha(self):
        return self._alpha
    
    @property
    def beta(self):
        return self._beta
    
    @property
    def delta(self):
        return self._delta
    
    @property
    def is_normalized(self):
        return self._is_normalized
    
    def __init__(self):
        """Initialize the polynomial."""
        pass
    
    def __call__(self, x):
        """Evaluate the polynomial basis at x."""
        x = np.atleast_2d(x).T # N x 1
        phi = np.zeros((x.shape[0], self.degree + 1)) # N x (P + 1)
        phi[0] = 1. / self.gamma[0]
        if self.degree >= 1:
            phi[1] = (x - self.alpha[0]) * (phi[0] / self.gamma[1])
        for i in range(2, self.degree + 1):
            phi[i] = ((x - self.alpha[i - 1]) * phi[i - 1] -
                self.beta[i - 1] * phi[i - 2]) / self.gamma[i]
        return phi