"""Define a real covariance function.

Author:
    Ilias Bilionis

Date:
    11/20/2012

"""


import numpy as np
from uq.gp import CovarianceFunction


class RealCovarianceFunction(CovarianceFunction):

    """Define a real covariance function.
    
    A real covariance function is a covariance function defined in R^n.

    """

    # The dimension of the space it lives in (imutable).
    _dim = None

    @property
    def dim(self):
        """Get the dimensions of the space."""
        return self._dim

    def __init__(self, dim, name="RealCovarianceFunction"):
        """Initialize the object.

        Arguments:
        dim     ---     The dimensions of the space.

        Keyword Arguments:
        name    ---     The name of the covariance function.

        """
        super(RealCovarianceFunction, self).__init__(name=name)
        if not isinstance(dim, int):
            raise TypeError('The number of dimensions must be an int.')
        if dim <= 0:
            raise ValueError('The number of dimensions must be positive.')
        self._dim = dim

    def _check_is_float(self, x, var_name):
        """Check if x is a float and self.dim is 1."""
        if isinstance(x, float):
            if not self.dim == 1:
                raise TypeError(var_name + ' must be a numpy array.')
            _x = np.ndarray(1)
            _x[0] = x
            return _x
        return x

    def _check_if_array_is_valid(self, x, var_name):
        """Check if x is a valid numpy array."""
        if not isinstance(x, np.ndarray):
            raise TypeError(var_name + ' must be a numpy array.')
        if len(x.shape) == 1:
            if not x.shape[0] == self.dim:
                raise TypeError(var_name + ' must have ' + self.dim
                                + ' dimensions.')
            return x.reshape((1, x.shape[0]))
        if not len(x.shape) == 2:
            raise TypeError(varm_name 
                            + ' cannot have more than two dimensions.')
        if not x.shape[1] == self.dim:
            raise TypeError(var_name + ' must have ' + self.dim
                            + ' elements in the second dimension.')
        return x

    def _eval(self, hyp, x1, x2, x2_is_x1):
        """Compute the covariance function at two given points.

        This function has to be implemented by the deriving classes.
    
        Arguments:
        hyp     ---     The hyper parameters.
        x1      ---     The first point.
        x2      ---     The second point.
        x2_is_x1---     Is the first point the same as the second point?

        """
        raise NotImplementedError(
                'The function _eval has to be implemented for a'
                + ' RealCovarianceFunction.')

    def __call__(self, hyp, x1, x2=None, A=None):
        """Compute the covariance function.

        See the docstring of CovarianceFunction for the definition of the
        arguments.

        """
        self._check_hyp(hyp)
        x1 = self._check_is_float(x1, 'x1')
        x1 = self._check_if_array_is_valid(x1, 'x1')
        if x2 is None or x2 is x1:
            x2 = x1
            x2_is_x1 = True
        else:
            x2 = self._check_is_float(x2, 'x2')
            x2 = self._check_if_array_is_valid(x2, 'x2')
            x2_is_x1 = False
        if A is None:
            A_is_None = True
            A = np.zeros((x1.shape[0], x2.shape[0]))
        else:
            A_is_None = False
        for i in xrange(x1.shape[0]):
            for j in xrange(x2.shape[0]):
                A[i, j] = self._eval(hyp, x1[i, :], x2[j, :], x2_is_x1)
        if A_is_None:
            return A

    def __str__(self):
        """Return a string representation of the object."""
        s = super(RealCovarianceFunction, self).__str__()
        s += '\ndim = ' + str(self.dim)
        return s
