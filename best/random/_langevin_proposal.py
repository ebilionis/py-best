"""An MCMC proposal emulating Langevin dynamics.

Author:
    Ilias Bilionis

Date:
    3/6/2013

"""


__all__ = ['LangevinProposal']


import numpy as np
import math
from . import ProposalDistribution


class LangevinProposal(ProposalDistribution):
    """A proposal emulating Langevin dynamics."""

    # The target distribution
    _target = None

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self):
        self._target = target

    def __init__(self, target, dt=1e-3):
        """Initialize the object.

        Arguments:
            target      ---     The target proposal.
        """
        super(LangevinProposal, self).__init__(dt=dt)
        self.target = target

    def __call__(self, x_p, x_n):
        """Evaluate the logarithm of the pdf of the chain."""
        k = x_p.shape[0]
        y = x_n - x_p
        dy = self.target.dlogp(x_p)
