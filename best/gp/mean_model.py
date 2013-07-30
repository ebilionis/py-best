"""The base class of all mean models.

Author:
    Ilias Bilionis

Date:
    12/2/2012

"""

import numpy as np


class MeanModel(object):
    """The base class of all mean models."""
    
    # Number of basis functions
    _m = None

    # Number of input dimensions
    _k = None

    @property
    def m(self):
        """Get the number of basis functions."""
        return self._m

    @property
    def k(self):
        """Get the number of input dimensions."""
        return self._k

    def __init__(self, k, m):
        """Initialize the object.
        
        Arguments:
            k   ---     The number of input dimensions.
            m   ---     The number of basis functions.

        """
        self._k = k
        self._m = m

    def __call__(self, X, H=None):
        """Calculate the design matrix."""
        raise NotImplementedError(
                'This function must be implemented by deriving classes.')
