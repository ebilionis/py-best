"""Define a Discrete Karhunen-Loeve Expansion.

Author:
    Ilias Bilionis

Date:
    12/13/2012

"""

import numpy as np


class DiscreteKarhunenLoeveExpansion(object):
    
    """Define a Discrete Karhunen-Loeve Expansion.
    
    This can also be used to represent a PCA expansion.

    """

    # The set of eigen vectors
    _PHI = None

    # The set of eigen-values
    _lam = None
    
    # The mean of the model
    _mean = None
    
    # The signal strength of the model
    _sigma = None

    @property
    def PHI(self):
        """Get the eigen vectors."""
        return self._PHI

    @property
    def lam(self):
        """Get the eigen values."""
        return self._lam
    
    @property
    def mean(self):
        """Get the mean of the model."""
        return self._mean
    
    @mean.setter
    def mean(self, value):
        """Set the mean of the model."""
        if not isinstance(value, np.ndarray):
            raise TypeError('mean must be a numpy array.')
        if not value.shape[0] == self.PHI.shape[0]:
            raise ValueError('The dimensions of mean and PHI do not agree.')
        if len(value.shape) == 2:
            value = value.reshape((value.shape[0],), order='F')
        self._mean = value
    
    @property
    def sigma(self):
        """Get the signal strength of the model."""
        return self._sigma
    
    @sigma.setter
    def sigma(self, value):
        """Set the signal strength of the model."""
        if not isinstance(value, float):
            raise TypeError('sigma must be a float.')
        if value < 0.:
            raise ValueError('sigma must be positive.')
        self._sigma = value
    
    @property
    def num_input(self):
        """Get the number of inputs."""
        return self._PHI.shape[1]

    @property
    def num_output(self):
        """Get the number of outputs."""
        return self._PHI.shape[0]

    def __init__(self, PHI, lam, mean=None, sigma=None):
        """Initialize the object.

        Arguments:
            PHI     ---     Should be the eigenvectors.
            lam     ---     Should be the eigenvalues.
            mean    ---     The mean of the model.
            sigma   ---     The signal strength of the model. The default value
                            is one.

        Precondition:
            PHI.shape[1] == lam.shape[1]

        """
        if not PHI.shape[1] == lam.shape[0]:
            raise ValueError('PHI and lam dimensions do not match.')
        self._PHI = PHI
        self._lam = lam
        if mean is None:
            mean = np.zeros(PHI.shape[0])
        self.mean = mean
        if sigma is None:
            sigma = 1.
        self.sigma = sigma

    def __call__(self, theta):
        """Evaluate the Discrete Karhunen-Loeve Expansion.

        Arguments:
            theta   ---     The KL weights. The dimension of theta must be
                            less than the dimension of _lambda. If it is
                            strictly, less then the remaining coefficients are
                            assumed to be zero.

        Return:
            The sample.

        """
        m = theta.shape[0]
        if self.PHI.shape[1] < m:
            raise ValueError('Theta has unknown dimensions.')
        return self._mean + np.dot(self.PHI[:, :m],
                                   self.sigma * np.sqrt(self.lam[:m]) * theta)
    
    def project(self, Y):
        """Project Y to the space of KL weights.
        
        Arguments:
            Y   ---     A sample of the field.
            
        Return:
            theta   ---     The KLE weights corresponding to Y.
        
        """
        return (np.dot(self.PHI.T, Y - self.mean) /
                (np.sqrt(self.lam) * self.sigma))
