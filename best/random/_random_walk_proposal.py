"""A random walk proposal distribution.

Author:
    Ilias Bilionis

Date:
    1/15/2013

"""


__all__ = ['RandomWalkProposal']


import numpy as np
import math
from . import ProposalDistribution


class RandomWalkProposal(ProposalDistribution):
    """A random walk proposal distribution."""

    def __init__(self, dt=1e-3, name='Random Walk Proposal'):
        """Initialize the object."""
        super(RandomWalkProposal, self).__init__(dt=dt, name=name)

    def __call__(self, x_p, x_n):
        """Evaluate the logarithm of the pdf of the chain."""
        k = x_p.shape[0]
        y = x_n - x_p
        return -0.5 * (k * math.log(2. * math.pi)
                       + k * math.log(self.dt)
                       + np.dot(y, y) / self.dt ** 2)

    def sample(self, x_p, x_n):
        """Sample from the pdf of the chain."""
        k = x_p.shape[0]
        x_n[:] = x_p + self.dt * np.random.randn(k)
