"""A class representing a mixture of distributions.

Author:
    Ilias Bilionis
    
Date:
    1/16/2013

"""

import numpy as np
import math
from uq.random import Distribution


class MixtureOfDistributions(Distribution):
    """A class representing a mixture of Gaussians.
    
    p(x) = \sum c_i p_i(x),
    
    where
    
    \sum c_i = 1,
    
    and p_i(x) are distributions.
    """
    
    # The logartihm of the weights of each component
    _log_w = None
    
    # The components (list of Distributions)
    _components = None
    
    @property
    def log_w(self):
        """Get the logarithm of the weights of each componet."""
        return self._log_w
    
    @property
    def weights(self):
        """Get the weights of each component."""
        return np.exp(self.log_w)
    
    @property
    def components(self):
        """Get the components."""
        return self._components
    
    @property
    def num_components(self):
        """Get the number of components."""
        return len(self.components)
    
    def __init__(self, weights, components, name='Mixture of Distributions'):
        """Initialize the object.
        
        Arguments:
        weights     ---     The weight of each component.
        components  ---     A list of the components. They need to
                            be NormalDistribution objects.

        Keyword Arguments:
        name        ---     Give a name to this distribution.
        """
        if not isinstance(weights, np.ndarray):
            raise TypeError('The weights must be a numpy array.')
        if not len(weights.shape) == 1:
            raise ValueError('The weights must be one-dimensional array.')
        # Compute the log of the weigths
        self._log_w = np.log(weights)
        # Make sure the weights are normalized
        self._log_w -= math.log(math.fsum(np.exp(self.log_w)))
        if not isinstance(components, list) and not isinstance(components, tuple):
            raise TypeError('The components must be a list or a tuple.')
        num_input = components[0].num_input
        super(MixtureOfDistributions, self).__init__(num_input, name=name)
        for c in components:
            if not isinstance(c, Distribution):
                raise TypeError('All the components must be Distributions')
            if not self.num_input == c.num_input:
                raise ValueError('The dimensions of each component must be the same.')
        if not weights.shape[0] == len(components):
            raise ValueError('The dimension of weights does not agree with the size of the component list.')
        self._components = components
    
    def __call__(self, x):
        """Evaluate the logarithm of the pdf at x."""
        log_p = np.array([c(x) for c in self.components])
        return math.log(math.fsum(np.exp(self.log_w + log_p)))
    
    def sample(self, x=None):
        """Sample the distribution."""
        i = np.random.multinomial(1, self.weights)
        if x is None:
            return self.components[i]()
        else:
            self.components[i](x)
