"""A likelihood function representing a Student-t distribution.

Author:
    Ilias Bilionis

Date:
    1/21/2013
"""


__all__ = ['StudentTLikelihoodFunction']


import numpy as np
import scipy
import math
from . import GaussianLikelihoodFunction


class StudentTLikelihoodFunction(GaussianLikelihoodFunction):
    """An object representing a Student-t likelihood function."""

    # The degrees of freedom
    _nu = None

    @property
    def nu(self):
        """Get the degrees of freedom."""
        return self._nu

    @nu.setter
    def nu(self, value):
        """Set the degrees of freedom."""
        if not isinstance(value, float):
            raise TypeError('nu must be a float.')
        self._nu = value

    def __init__(self, nu, num_input=None, data=None, mean_function=None, cov=None,
                 name='Student-t Likelihood Function'):
        """Initialize the object.

        Arguments:
            nu                  ---     The degrees of freedom of the distribution.

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
        self.nu = nu
        super(StudentTLikelihoodFunction, self).__init__(num_input=num_input,
                                                         data=data,
                                                         mean_function=mean_function,
                                                         cov=cov,
                                                         name=name)

    def __call__(self, x):
        """Evaluate the function at x."""
        mu = self.mean_function(x)
        y = scipy.linalg.solve_triangular(self.L_cov, self.data - mu)
        return (
                - 0.5 * (self.nu + self.num_data) * math.log(1. + np.dot(y, y) / self.nu))
