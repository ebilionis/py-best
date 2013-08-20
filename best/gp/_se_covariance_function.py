"""Define the Squared Exponential Covariance Function

Author:
    Ilias Bilionis

Date:
    11/20/2012

"""


__all__ = ['SECovarianceFunction']


import numpy as np
import math
from . import RealCovarianceFunction



class SECovarianceFunction(RealCovarianceFunction):

    """A Squared Exponential Covariance Function."""

    def __init__(self, dim, name="Squared Exponential Covariance"):
        """Initialize the object.

        For the arguments see the docstring of RealCovarianceFunction.

        """
        super(SECovarianceFunction, self).__init__(dim, name=name)

    def _check_hyp(self, hyp):
        """Check if the hyperparameters are valid.

        They are assumed to be a numpy array of the right dimensions.
        They should also be positive, but we don't check for this for
        computational efficiency reasons.

        """
        if not isinstance(hyp, np.ndarray):
            raise TypeError(
                    'The hyperparameters of the SE covariance must be a numpy'
                    + ' array.')
        if not len(hyp.shape) == 1:
            raise TypeError(
                    'The hyperparameters must be a one-dimensional array.')
        if not hyp.shape[0] == self.dim:
            raise TypeError(
                    'The hyperparameters must be of dimension ' + str(self.dim)
                    + '.')

    def _eval(self, hyp, x1, x2, x2_is_x1):
        """Evaluate the covariance function at two points.

        See the docstring of RealCovarianceFunction for more details.

        """
        diff = (x1 - x2) / hyp
        return math.exp(-0.5 * np.dot(diff, diff))

    def __call__(self, hyp, x1, x2=None, A=None):
        if x2 is None:
            x2 = x1
        if A is None:
            A = np.ndarray((x1.shape[0], x2.shape[0]))
            is_A_None = True
        else:
            is_A_None = False
        A[:] = np.exp( - 0.5 *
            np.sum(
                (np.tile(x1 / (hyp), (x2.shape[0], 1))
                - np.repeat(x2 / (hyp), x1.shape[0], axis=0)
                ) ** 2,
                    axis=1)).reshape(x1.shape[0], x2.shape[0], order='F')
        if is_A_None:
            return A

    def d(self, hyp, x1, x2, A=None, J=None):
        """Calculate the derivative of a covariance matrix."""
        assert x2.shape[0] == 1
        if A is None:
            A = self(hyp, x1, x2)
        A = A.reshape((x1.shape[0], ))
        B = ((x1 - np.repeat(x2, x1.shape[0], axis=0)) / (hyp ** 2)).T
        is_J_None = False
        if J is None:
            is_J_None = True
            J = np.ndarray((x1.shape[1], x1.shape[0]))
        J[:] = B * A
        if is_J_None:
            return J