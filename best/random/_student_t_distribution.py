"""Define a Student T distribution.

Author:
    Ilias Bilionis

Date:
    1/14/2013

"""


__all__ = ['StudentTDistribution']


import numpy as np
import math
from . import NormalDistribution


class StudentTDistribution(NormalDistribution):
    """A student-t distribution."""

    # The degrees of freedom
    _nu = None

    @property
    def nu(self):
        """Get the degrees of freedom."""
        return self._nu

    @nu.setter
    def nu(self, value):
        """Set the degrees of freedom."""
        if not isinstance(value, float):
            raise TypeError('nu must be a float.')
        self._nu = value

    def __init__(self, num_input, nu, mu=None, cov=None,
            name='Student-t Distribution'):
        """Initialize the object.

        Arguments:
        num_input   ---     The dimension of the random variables.
        nu  ---     The degrees of freedom.

        Keyword Arguments:
        mu      ---     The mean.
        cov     ---     The covariance.
        name    ---     A name for the distribution.
        """
        super(StudentTDistribution, self).__init__(num_input, mu=mu, cov=cov,
                name=name)
        self.nu = nu

    def __call__(self, x):
        """Evaluate the logarithm of the pdf at x."""
        y = scipy.linalg.solve_triangular(self.L_cov, x - self.mu)
        return (math.log(math.gamma(0.5 * (self.nu + self.num_input)))
                - math.log(math.gamma(0.5 * self.nu))
                - 0.5 * self.num_input * math.log(self.nu)
                - 0.5 * self.num_input * math.log(math.pi)
                - 0.5 * self.log_det_cov
                - 0.5 * (self.nu + self.num_input) * math.log(1. + np.dot(y, y) / self.nu))

    def sample(self, x=None):
        """Sample the distribution."""
        z = np.random.standard_t(self.nu, size=self.num_input)
        y = self.mu + np.dot(self.L_cov, z)
        if x is None:
            return y
        else:
            x[:] = y