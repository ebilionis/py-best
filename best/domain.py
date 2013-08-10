"""A class that represents a domain of a Eucledian space.

Author:
    Ilias Bilionis

Date:
    8/11/2013
"""

import numpy as np


class Domain(object):

    """A class that represents a domain of a Euclidean space."""

    # The number of dimensions
    _num_dim = None

    # A name for the domain
    _name = None

    @property
    def num_dim(self):
        return self._num_dim

    @property
    def name(self):
        return self._name

    def __init__(self, num_dim, name='Domain'):
        """Initialize the object.

        Arguments:
            num_dim     ---     The number of dimensions.

        Keyword arguments
            name        ---     A name for the domain.
        """
        assert isinstance(num_dim, int)
        assert num_dim >= 0
        self._num_dim = num_dim
        assert isinstance(name, str)
        self._name = name

    def is_in(self, x):
        """Return True if x is in the domain.

        Note:   This must be implemented by the children of this class.
        """
        raise NotImplementedError('Must be implemented by children.')

    def __str__(self):
        """Return a string representation of the object."""
        return 'Domain: ' + self.name + ' < R^' + str(self.num_dim)


class DomainRectangle(Domain):

    """Represents a rectangular domain."""

    # The rectangle
    _rectangle = None

    @property
    def rectangle(self):
        return self._rectangle

    def __init__(self, rectangle, name='Rectangular Domain'):
        """Initialize the object.

        Arguments:
            rectangle       ---     This must be an num_dim x 2 array
                                    representing the rectangle.
        """
        if not isinstance(rectangle, np.ndarray):
            rectangle = np.array(rectangle)
        rectangle = np.atleast_2d(rectangle)
        assert len(rectangle.shape) == 2
        assert rectangle.shape[1] == 2
        # Check consistency
        assert (rectangle[:, 0] <= rectangle[:, 1]).all()
        self._rectangle = rectangle
        num_dim = rectangle.shape[0]
        super(DomainRectangle, self).__init__(num_dim, name=name)

    def is_in(self, x):
        """Test if x is in the domain."""
        return ((self.rectangle[:, 0] <= x).all()
                and (x <= self.rectangle[:, 1]).all())

    def __str__(self):
        s = super(DomainRectangle, self).__str__()
        s += '\nRectangle: ' + str(self.rectangle)
        return s


class DomainUnitCube(DomainRectangle):

    """A unit cube domain."""

    def __init__(self, num_dim, name='Unit Cube Domain'):
        """Initialize the object.

        Arguments:
            num_dim     ---     The number of dimensions.

        Keyword Arguments
            name        ---     A name for the domain.
        """
        assert isinstance(num_dim, int)
        assert num_dim >= 1
        rectangle = np.zeros((num_dim, 2))
        rectangle[:, 1] = 1.
        super(DomainUnitCube, self).__init__(rectangle, name=name)


class DomainAllSpace(DomainRectangle):

    """A domain representing all space."""

    def __init__(self, num_dim, name='All Space Domain'):
        """Initialize the object.

        Arguments:
            num_dim     ---     The number of dimensions.

        Keyword Arguments
            name        ---     A name for the domain.
        """
        assert isinstance(num_dim, int)
        assert num_dim >= 1
        rectangle = np.zeros((num_dim, 2))
        rectangle[:, 0] = -float('inf')
        rectangle[:, 1] = float('inf')
        super(DomainAllSpace, self).__init__(rectangle, name=name)


if __name__ == '__main__':
    d = DomainUnitCube(5)
    print str(d)
    x = np.ones(5) * 1.4
    print d.is_in(x)