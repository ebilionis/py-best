"""The base class for proposal distributions used in MCMC.

Author:
    Ilias Bilionis
    
Date:
    1/15/2013
    
"""

from uq.random import MarkovChain


class ProposalDistribution(MarkovChain):
    """The base class for proposals used in MCMC."""
    
    # The step size
    _dt = None
    
    @property
    def dt(self):
        """Get the step size."""
        return self._dt
    
    @dt.setter
    def dt(self, value):
        """Set the step size."""
        if not isinstance(value, float):
            raise TypeError('The step size must be a float.')
        if value <= 0.:
            raise ValueError('The step size must be positive.')
        self._dt = value
    
    def __init__(self, dt=1e-3):
        """Initialize the object.
        
        Keyword Arguments:
        dt  ---     The step size of the proposal.
        """
        super(ProposalDistribution, self).__init__()
        self.dt = dt