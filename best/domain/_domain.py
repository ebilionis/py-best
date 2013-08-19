"""A class that represents a domain of a Eucledian space.

Author:
    Ilias Bilionis

Date:
    8/11/2013
"""


__all__ = ['Domain', 'Rectangle', 'UnitCube', 'AllSpace']


import numpy as np
import best


class Domain(best.Object):

    """A class that represents a domain of a Euclidean space."""

    # The number of dimensions
    _num_dim = None

    @property
    def num_dim(self):
        return self._num_dim

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
        super(Domain, self).__init__(name=name)

    def is_in(self, x):
        """Return True if x is in the domain.

        Note:   This must be implemented by the children of this class.
        """
        raise NotImplementedError('Must be implemented by children.')

    def _to_string(self, pad):
        """Return a string representation of the object."""
        s = super(Domain, self)._to_string(pad) + '\n'
        s += pad + ' Subset of R^' + str(self.num_dim)
        return s


class Rectangle(Domain):

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
        super(Rectangle, self).__init__(num_dim, name=name)

    def is_in(self, x):
        """Test if x is in the domain."""
        return ((self.rectangle[:, 0] <= x).all()
                and (x <= self.rectangle[:, 1]).all())

    def _to_string(self, pad):
        """Return a string representation of the object."""
        s = super(Rectangle, self)._to_string(pad) + '\n'
        s += pad + ' box: ' + str(self.rectangle)
        return s


class UnitCube(Rectangle):

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
        super(UnitCube, self).__init__(rectangle, name=name)


class AllSpace(Rectangle):

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
        super(AllSpace, self).__init__(rectangle, name=name)


if __name__ == '__main__':
    d = UnitCube(5)
    print str(d)
    x = np.ones(5) * 1.4
    print d.is_in(x)