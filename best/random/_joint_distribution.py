"""The joint distribution of two random variables.

Author:
    Ilias Bilionis

Date:
    2/21/2013

"""


__all__ = ['JointDistribution']


import numpy as np
from . import Distribution


class JointDistribution(Distribution):
    """The joint distribution of a bunch of random variables."""

    # The collection of underlying distributions (list)
    _dist = None

    @property
    def dist(self):
        """Get the underlying distributions."""
        return self._dist

    def __init__(self, dist, name='Joint distribution'):
        """Initialize the object.

        Arguments:
            dist        ---     A list of distributions to be joined.

        Keyword Arguments:
            name        ---     A name for the distribution.
        """
        for d in dist:
            assert isinstance(d, Distribution)
        num_input = np.prod([d.num_input for d in dist])
        self._dist = dist
        super(JointDistribution, self).__init__(num_input, name=name)

    def __call__(self, x):
        """Evaluate the logarithm of the pdf at x."""
        r = 0.
        start = 0.
        for i in range(len(self.dist)):
            end = start + self.dist[i].num_input
            r += self.dist[i](x[start:end])
            start = end
        return r

    def sample(self, x=None):
        """Sample the distribution."""
        if x is None:
            y = np.zeros(self.num_input)
        else:
            y = x
        start = 0
        for i in range(len(self.dist)):
            end = start + self.dist[i].num_input
            self.dist[i](x[start:end])
            start = end
        if x is None:
            return y