"""The base class of a surrogate model.

Author:
    Ilias Bilionis

Date:
    11/20/2012

"""


class SurrogateModel(object):

    """A class that represents a surrogate of a computer code."""

    # Number of inputs (imutable)
    _num_input = None

    # Number of outputs (imutable)
    _num_output = None

    # Number of spatial variables (imutable)
    _num_spatial = None

    # Does it evolve in time? (imutable)
    _has_time = None

    # A name for the surrogate
    _name = None

    @property
    def num_input(self):
        """Return the number of inputs."""
        return self._num_input

    @property
    def num_output(self):
        """Return the number of outputs."""
        return self._num_output

    @property
    def num_spatial(self):
        """Return the number of spatial inputs."""
        return self._num_spatial

    @property
    def has_spatial(self):
        """Does it evolve in space?"""
        return self._num_spatial > 0

    @property
    def has_time(self):
        """Does it evolve in time?"""
        return self._has_time

    @property
    def name(self):
        """Get the name of the surrogate."""
        return self._name

    @name.setter
    def name(self, value):
        """Set the name of the surrogate."""
        if not isinstance(value, str):
            raise TypeError(
                    'The name of the surrogate must be a string.')
        self._name = value

    def __init__(self, num_input, num_output, num_spatial=0, has_time=False,
                 name='SurrogateModel'):
        """Initialize the object.

        Arguments:
        num_input   ---     The number of inputs.
        num_output  ---     The number of outputs.

        Keyword Arguments:
        num_spatial ---     The number of spatial dimensions.
        has_time    ---     Does it evolve in time?
        name        ---     A name for the surrogate.
        
        """
        if not isinstance(num_input, int):
            raise TypeError('num_input not an int.')
        if num_input < 0:
            raise ValueError('num_input < 0.')
        self._num_input = num_input
        if not isinstance(num_output, int):
            raise TypeError('num_output not an int.')
        if num_output < 0:
            raise ValueError('num_output < 0.')
        self._num_output = num_output
        if not isinstance(num_spatial, int):
            raise TypeError('num_spatial not an int.')
        if num_spatial < 0 or num_spatial > 3:
            raise ValueError('num_spatial must be 0, 1, 2 or 3.')
        self._num_spatial = num_spatial
        if not isinstance(has_time, bool):
            raise TypeError('has_time must be a boolean.')
        self._has_time = has_time
        self.name = name

    def __str__(self):
        """Return a string representation of the object."""
        s = self.name
        s += '\nnum_input = ' + str(self.num_input)
        s += '\nnum_output = ' + str(self.num_output)
        if self.has_spatial:
            s += '\nnum_spatial = ' + str(self.num_spatial)
        if self.has_time:
            s += '\nhas_time'
        return s

    def _check_if_X(self, X):
        """Check if X is compatible with the object."""
        pass

    def __call__(self, X, Y=None, C=None):
        """Evaluate the code.

        This should obviously be reimplemented by deriving classes.

        Arguments:
        X   ---     The general input. We have the following cases:
                        + self.has_spatial and self.has_time are False. Then, X
                        should be either of the same type as the input, or a
                        tuple with as single element.
                        + self.has_spatial is False but self.has_time is Frue.
                        Then X MUST be a tuple. X[0] should be the inputs and
                        X[1], the time.
                        + self.has_spatial is True but self.has_time is False.
                        Then X MUST be a tuple. X[0] should be the inputs and
                        X[1], the spatial variables.
                        + self.has_spatial is True and self.has_time is True.
                        Then X MUST be a tuple. X[0] should be the inputs, X[1]
                        should the spatial variables and X[2] should be the
                        time variable.

        Keyword Argument:
        Y   ---     The output of the surrogate. This should be the output an
                    array organized storing the output on the tensor product of
                    X[0], X[1] and X[2] for each output. If None, then the
                    output should be allocated and returned.
        C   ---     If C is not None, then this function should calculate the
                    covariance matrix of all outputs.
        
        """
        raise NontImplemented(
            'The __call__ function is not implemented.')
