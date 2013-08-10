"""A class representing a random vector.

Author:
    Ilias Bilionis

Date:
    8/11/2013
"""

import numpy as np
import scipy.stats as stats


class RandomVector(object):

    """A class representing a random vector.

    The purpose of this class is to serve as a generalization of
    scipy.stats.rv_continuous. It should offer pretty much the same
    functionality.
    """

    # The support of the random variable
    _support = None

    # A name for the random vector
    _name = None

    # The number of dimensions
    _num_dim = None

    @property
    def support(self):
        return self._support

    @property
    def num_dim(self):
        return self._num_dim

    @property
    def name(self):
        return self._name

    def __init__(self, num_dim, support=None, name='Random Vector'):
        """Initialize the object.

        Arguments:
            num_dim     ---     The number of dimensions of the random
                                vector.

        Keyword Arguments:
            support     ---     The support of the random vector. If
                                None, then it is assume to be all space.
            name        ---     A name for the random vector.
        """
        assert isinstance(num_dim, int)
        assert num_dim > 1
        if support is None:
            support = np.ndarray([self.num_dim, 2])
            support[:, 0] = -float('inf')
            support[:, 1] = float('inf')
        self._support = support
        assert isinstance(name, str)
        self._name = name

    def _is_in_support(self, x):
        """Return True if the random variable is in the support."""
        return (support[:, 0] >= x).all() and (x <= support[:, 1]).all()
