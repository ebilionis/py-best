"""A class that implements a generalized linear model.

Author:
    Ilias Bilionis

Date:
    8/14/2013
"""


import numpy as np
from function import Function


class GeneralizedLinearModel(Function):

    """A class that implements a generalized linear model."""

    # The basis (must be a best.maps.Function)
    _basis = None

    # The weight matrix/vector
    _weights = None

    # The square root of the covariance matrix.
    _sigma_sqrt = None

    # The noise precision
    _beta = None

    @property
    def basis(self):
        return self._basis

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, val):
        val = np.array(val)
        val = np.atleast_2d(val)
        if self.basis.num_output == 1 and not val.shape[0] == 1:
            val = val.T
        if val.shape[1] == 1:
            val = val.reshape((val.shape[0], ))
        self._weights = val

    @property
    def sigma_sqrt(self):
        return self._sigma_sqrt

    @sigma_sqrt.setter
    def sigma_sqrt(self, val):
        val = np.array(val)
        assert val.ndim == 2
        assert val.shape[1] == self.basis.num_output
        self._sigma_sqrt = val

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, val):
        val = float(val)
        assert val > 0.
        self._beta = val

    def __init__(self, basis, weights=None, sigma_sqrt=None,
                 beta=1e99,
                 num_output=1,
                 name='Mean of Generalized Linear Model'):
        """Initialize the object.

        Arguments:
            basis       ---     A basis for the model. It must be a
                                best.maps.Function.

        Keword Arguments:
            weights     ---     The weight matrix vector. It is a weight
                                matrix if basis.num_output > 1. If None,
                                then it is initialized to zero and its
                                dimensionality is found by looking at
                                num_output. It should be a
                                self.num_output x self.basis.num_output
                                matrix.
            sigma_sqrt  ---     The square root of the covariance
                                function.
            beta        ---     The noise precision.
            num_output  ---     The number of output dimensions of the
                                generalized linear model. If weights is
                                not None, then it is ignored.
            name        ---     A name for the model.
        """
        assert isinstance(basis, Function)
        self._basis = basis
        if weights is None:
            weights = np.zeros((basis.num_output, num_output))
        if sigma_sqrt is None:
            sigma_sqrt = np.zeros((basis.num_output, basis.num_output))
        self.sigma_sqrt = sigma_sqrt
        self.beta = beta
        super(GeneralizedLinearModel, self).__init__(self.basis.num_input,
                                                     num_output,
                                                     name=name)
        self.weights = weights

    def __call__(self, x):
        """Evaluate the mean at x."""
        return np.dot(self.basis(x), self.weights)

    def d(self, x):
        """Evaluate the derivative of the mean at x."""
        return np.dot(self.basis.d(x), self.weights)

    def get_predictive_covariance(self, x):
        """Evaluate the predictive covariance matrix at x."""
        tmp = np.dot(self.sigma_sqrt, self.basis(x).T)
        return np.dot(tmp.T, tmp) + np.eye(tmp.shape[1]) / self.beta

    def get_predictive_variance(self, x):
        """Evaluate the predictive variance at x."""
        D = np.dot(self.sigma_sqrt, self.basis(x).T)
        return (np.einsum('ij, ij->j', D, D) + np.ones(self.num_output)
                / self.beta)