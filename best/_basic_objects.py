"""Defines the basic objects of best.

Author:
    Ilias Bilionis

Date:
    8/19/2013
"""


__all__ = ['Object']


class Object(object):

    """A basic best object.

    It is defined to be a python object with a name.
    Every BEST object inherits from Object.
    """

    # The name of the object.
    _name = None

    @property
    def name(self):
        """Get the name of the object."""
        return self._name

    def __init__(self, name='Object'):
        """Initialize the object.

        Keyword Arguments:
            name    ---     A name for the object.
        """
        if not isinstance(name, str):
            raise TypeError('A name must be a string.')
        self._name = name

    def _to_string(self, pad):
        """Return a string representation of the object.

        This is the method that should be overloaded by deriving classes.

        Arguments:
            pad     ---     A padding that goes in front of the string.
        """
        s = pad + 'Best Object\n'
        s = pad + str(self.name)
        return s

    def __str__(self):
        """Return a string representation of the object."""
        s = self._to_string('')
        return s