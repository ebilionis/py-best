"""A likelihood function representing a Gaussian.

Author:
    Ilias Bilionis

Date:
    1/21/2013
"""

import numpy as np
import scipy
import math
from uq.random import LikelihoodFunctionWithGivenMean


class GaussianLikelihoodFunction(LikelihoodFunctionWithGivenMean):
    """The class represents a Gaussian likelihood function.
    
    Namely:
        p(D | x) = N(D| f(x), C),
    
    where C is the covariance matrix.
    """
    
    # The covariance matrix
    _cov = None
    
    # The Cholesky decomposition of the covariance matrix
    _L_cov = None
    
    # The logarithm of the determinant of the covariance matrix
    _log_det_cov = None
    
    @property
    def cov(self):
        """Get the covariance matrix."""
        return self._cov
    
    @cov.setter
    def cov(self, value):
        """Set the covariance matrix."""
        if isinstance(value, float) or isinstance(value, int):
            value = np.eye(self.num_data) * float(value)
        if not isinstance(value, np.ndarray):
            raise TypeError('The covariance must a numpy array.')
        if (len(value.shape) != 2
            or (len(value.shape) == 2 and value.shape[0] != value.shape[1])):
            raise ValueError('The covariance matrix must be square.')
        self._cov = value
        self._L_cov = np.linalg.cholesky(self.cov)
        self._log_det_cov = 2. * np.log(np.diag(self.L_cov)).sum()
    
    @property
    def L_cov(self):
        """Get the Cholesky decomposition of the covariance."""
        return self._L_cov
    
    @property
    def log_det_cov(self):
        """Get the log of the det of the covariance."""
        return self._log_det_cov
    
    def __init__(self, num_input=None, data=None, mean_function=None, cov=None,
                 name='Gaussian Likelihood Function'):
        """Initialize the object.
        
        Keyword Arguments
            num_input           ---     The number of inputs. Optional, if
                                        mean_function is a proper Function.
            data                ---     The observed data. A vector. Optional,
                                        if mean_function is a proper Function.
                                        It can be set later.
            mean_function       ---     The mean function. See the super class
                                        for the description.
            cov                 ---     The covariance matrix. It can either be
                                        a positive definite matrix, or a number.
                                        The data or a proper mean_funciton is
                                        preassumed.
            name                ---     A name for the likelihood function.
        """
        super(GaussianLikelihoodFunction, self).__init__(num_input=num_input,
                                                         data=data,
                                                         mean_function=mean_function,
                                                         name=name)
        if cov is not None:
            self.cov = cov
    
    def __call__(self, x):
        """Evaluate the function at x."""
        mu = self.mean_function(x)
        y = scipy.linalg.solve_triangular(self.L_cov, self.data - mu)
        return (-0.5 * self.num_data * math.log(2. * math.pi)
                -0.5 * self.log_det_cov
                -0.5 * np.dot(y, y))