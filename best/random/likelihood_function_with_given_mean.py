"""Defines a likelihood that has a mean that is a function.

Author:
    Ilias Bilionis

Date:
    1/21/2013
"""

import numpy as np
from uq.random import LikelihoodFunction
from uq.maps import Function


class LikelihoodFunctionWithGivenMean(LikelihoodFunction):
    """This class implements a likelihood function, that requires the evaluation
    of another function within it (.e.g. that of a forward solver.). It is not
    to be used by its own, but rather as a base class for more specific likelihood
    functions.
    
    Here, we assume that the data are actually a num_data vector and that the
    mean of the likelihood is given by a function (mean_function) with num_input
    variables to num_data variables.
    """
    
    # The mean function
    _mean_function = None
    
    @property
    def data(self):
        """Get the data."""
        return super(LikelihoodFunctionWithGivenMean, self).data
    
    @data.setter
    def data(self, value):
        """Set the data (only accept numpy arrays)."""
        if not isinstance(value, np.ndarray):
            raise TypeError('The data must be a numpy array.')
        if not len(value.shape) == 1:
            raise ValueError('The data must be a vector.')
        self._data = value
    
    @property
    def num_data(self):
        """Get the number of dimensions of the data."""
        if self.data is not None:
            return self.data.shape[0]
        elif self.mean_function is not None:
            return self.mean_function.num_output
        else:
            raise RuntimeError(
                'Either data or the mean_function must be specified.')

    @property
    def mean_function(self):
        """Get the mean function."""
        return self._mean_function

    @mean_function.setter
    def mean_function(self, value):
        """Set the mean function."""
        if not isinstance(value, Function) and hasattr(value, '__call__'):
            value = Function(self.num_input, self.num_data, f_wrapped=value)
        if not isinstance(value, Function):
            raise TypeError('The mean function must be a function.')
        self._mean_function = value

    def __init__(self, num_input=None, data=None, mean_function=None,
                 name='Likelihood Function with Given Mean'):
        """Initializes the object.

        Careful:
            Either num_input or mean_function must be specified.
            If mean_function is a simple function, then the data are required
            so that we can figure out its output dimension.

        Keyword Arguments:
            num_input       ---     Number of input dimensions. Ignored, if
                                    mean_function is a Function class.
            data            ---     The observed data. A numpy vector. It must
                                    be specified if mean_function is a normal
                                    function.
            mean_function   ---     A Function or a normal function. If, it is
                                    a Function, then mean_function.num_output
                                    must be equal to data.shape[0]. If it is
                                    a normal function, then it is assumed that
                                    it returns a data.shape[0]-dimensional vector.
            name            ---     A name for the likelihood function.
        """
        if mean_function is not None and isinstance(mean_function, Function):
            num_input = mean_function.num_input
        if num_input is None:
            raise ValueError('The number of input dimensions must be specified.')
        super(LikelihoodFunctionWithGivenMean, self).__init__(num_input, data=data,
                                                              name=name)
        if mean_function is not None:
            self.mean_function = mean_function

    def _to_string(self, pad):
        """Return a string representation of the object."""
        s = super(LikelihoodFunctionWithGivenMean, self)._to_string(pad)
        s += '\n' + pad + 'Mean Function:\n'
        s += self.mean_function._to_string(pad + ' ')
        return s
