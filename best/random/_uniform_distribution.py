"""Define the Uniform distribution on a square domain.

Author:
    Ilias Bilionis

Date:
    1/14/2013

"""


__all__ = ['UniformDistribution']


import numpy as np
from . import Distribution


class UniformDistribution(Distribution):
    """The Uniform distribution on a square domain."""

    # The domain of the random variables(k x 2 matrix)
    _domain = None

    @property
    def domain(self):
        """Get the domain of the random variables."""
        return self._domain

    @domain.setter
    def domain(self, value):
        """Set the domain of the random variables."""
        if not isinstance(value, np.ndarray):
            raise TypeError('The domain must be a numpy array.')
        if (not len(value.shape) == 2 or not value.shape[0] == self.num_input
            or not value.shape[1] == 2):
            raise ValueError('The domain must be (k x 2) dimensional.')
        for i in range(self.num_input):
            if not value[i, 0] <= value[i, 1]:
                raise ValueError('Domain error: left > right.')
        self._domain = value

    def __init__(self, num_input, domain=None, name='Uniform Distribution'):
        """Initialize the object.

        Arguments:
        k   ---     The number of dimensions.

        Keyword Arguments:
        domain  ---     The domain of the random variables. Must be a (k x 2)
                        matrix. If not specified, then a unit hyper-cube is
                        used.
        name    ---     A name for this distribution.
        """
        super(UniformDistribution, self).__init__(num_input, name=name)
        if domain is None:
            domain = np.zeros((self.num_input, 2))
            domain[:, 1].fill(1.)
        self.domain = domain

    def is_in_domain(self, x):
        for i in xrange(self.num_input):
            if x[i] < self.domain[i, 0] or x[i] > self.domain[i, 1]:
                return False
        return True

    def __call__(self, x):
        """Evaluate the logartihm of the pdf at x."""
        if self.is_in_domain(x):
            return 0.
        else:
            return -1e99

    def sample(self, x=None):
        """Sample the distribution."""
        y = np.random.rand(self.num_input)
        y = self.domain[:, 0] + (self.domain[:, 1] - self.domain[:, 0]) * y
        if x is None:
            return y
        else:
            x[:] = y
