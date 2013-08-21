"""Define a normal distribution.

Author:
    Ilias Bilionis

Date:
    1/14/2013

"""


__all__ = ['NormalDistribution']


import numpy as np
import scipy
import math
from . import Distribution


class NormalDistribution(Distribution):
    """Define a normal distribution."""

    # The mean k-dim array
    _mu = None

    # The covariance matrix (k x k), symmetric positive definite.
    _cov = None

    # The Cholesky decomposition of the covariance.
    _L_cov = None

    # The determinant of the covariance
    _log_det_cov = None

    @property
    def mu(self):
        """Get the mean."""
        return self._mu

    @mu.setter
    def mu(self, value):
        """Set the mean."""
        if not isinstance(value, np.ndarray):
            raise TypeError('Mu must be a numpy array.')
        if not len(value.shape) == 1 or not value.shape[0] == self.num_input:
            raise ValueError('Mu must be k-dimensional.')
        self._mu = value

    @property
    def cov(self):
        """Get the covariance matrix."""
        return self._cov

    @cov.setter
    def cov(self, value):
        """Set the covariance."""
        if isinstance(value, float):
            if value <= 0.:
                raise ValueError('The variance must be positive.')
            value = np.eye(self.num_input) * value
        if not isinstance(value, np.ndarray):
            raise TypeError('The covariance must be a numpy array.')
        if (not len(value.shape) == 2 or not value.shape[0] == self.num_input
            or not value.shape[1] == self.num_input):
            raise ValueError('The covariance must be a k x k matrix.')
        self._cov = value
        self._L_cov = np.linalg.cholesky(self.cov)
        self._log_det_cov = 2. * np.log(np.diag(self.L_cov)).sum()

    @property
    def L_cov(self):
        """Get the Cholesky decomposition of the covariance."""
        return self._L_cov

    @property
    def log_det_cov(self):
        """Get the determinant of the covariance."""
        return self._log_det_cov

    def __init__(self, num_input, mu=None, cov=None,
            name='Normal Distribution'):
        """Initialize the object.

        Arguments:
        num_input   ---     The dimension of the random variables.

        Keyword Arguments:
        mu      ---     The mean. Zero if not specified.
        cov     ---     The covariance matrix. Unit matrix if not specified.
        name    ---     A name for the distribution.
        """
        super(NormalDistribution, self).__init__(num_input, name=name)
        if mu is None:
            mu = np.zeros(self.num_input)
        self.mu = mu
        if cov is None:
            cov = 1.
        self.cov = cov

    def __call__(self, x):
        """Evaluate the logarithm of the pdf at x."""
        assert x.shape[0] == self.num_input
        y = scipy.linalg.solve_triangular(self.L_cov, x - self.mu)
        return (-0.5 * self.num_input * math.log(2. * math.pi)
                -0.5 * self.log_det_cov
                -0.5 * np.dot(y, y))

    def sample(self, x=None):
        """Sample the distribution."""
        z = np.random.randn(self.num_input)
        y = self.mu + np.dot(self.L_cov, z)
        if x is None:
            return y
        else:
            x[:] = y