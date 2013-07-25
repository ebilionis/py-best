"""An SPGP random element.

Author:
    Ilias Bilionis
    
Date:
    4/1/2013
"""

from uq import BinaryTree


class SPGPElement(BinaryTree):
    """Assuming that this element is based on [0, 1]^d."""
    
    # Number of dimensions
    _num_dim = None
    
    # Splitting dimension
    _split_dim = None
    
    # Splitting point
    _split_pt = None
    
    # MCMC Sampler
    _mcmc_sampler = None
    
    @property
    def num_dim(self):
        return self._num_dim
    
    @property
    def split_dim(self):
        return self._split_dim
    
    @property
    def split_pt(self):
        return self._split_pt
    
    @property
    def mcmc_sampler(self):
        return self._mcmc_sampler
    
    def __init__(self):
        self._num_dim = X.shape[0]
        super(SPGPElement, self).__init__()

    def train(self, X, Y, num_samples):
        pass