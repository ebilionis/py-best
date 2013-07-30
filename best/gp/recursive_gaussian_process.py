"""A recursvie version of Gaussian process regression.

Author:
    Ilias Bilionis
    
Date:
    12/27/2012

"""

import numpy as np
from uq.gp import SECovarianceFunction

class RecursiveGaussianProcess(object):
    
    """A recursive version of Gaussian process regression."""
    
    # Input data
    _X = None
    
    # Output data
    _Y = None
    
    # Design matrix
    _H = None
    
    # The covariance function
    _cov = None
    
    # The hyper-parameters
    _hyp = None
    
    @property
    def X(self):
        """Get the input data."""
        return self._X
    
    @property
    def Y(self):
        """Get the output data."""
        return self._Y
    
    @property
    def H(self):
        """Get the design matrix."""
        return self._H
    
    @property
    def cov(self):
        """Get the covariance function."""
        return self._cov
    
    @property
    def hyp(self):
        """Get the hyper-parameters."""
        return self._hyp
    
    def __init__(self):
        """Initialize the object."""
        pass
    
    def set_data(self, X, H, Y):
        """Set the observed data."""
        self._X = X
        self._H = H
        self._Y = Y
        self._cov = SECovarianceFunction(X.shape[1])
        self._hyp = [np.ndarray(X.shape[1]), 1e-6]
        self._hyp[0].fill(0.1)
    
    def compute_log_likelihood(self, hyp):
        """Compute the log likelihood of the data."""
        
    
    def __call__(self, X, H, Y=None, C=None):
        """Evaluate the regression function."""
        pass