"""Models a conditional distribution.

    p(z | x) = p(z, x) / p(x).

Author:
    Ilias Bilionis

Date:
    2/21/2013

"""

from uq.random import Distribution


class ConditionalDistribution(Distribution):
    """Base class for conditional distributions."""

    # Number of conditioning variables
    _num_cond = None

    @property
    def num_cond(self):
        """Get the number of conditioning variables."""
        return self._num_cond

    def __init__(self, num_input, num_cond, name='Conditional Distribution.'):
        """Initialize the object.

        Arguments:
            num_input       ---     Number of inputs.
            num_cond        ---     Number of conditioning variables.
        """
        super(ConditionalDistribution, self).__init__(num_input,
                name=name)
        assert isinstance(num_cond, int)
        assert num_cond >= 0
        self._num_cond = num_cond

    def __call__(self, z, x):
        """Evaluate the log probability at z given x."""
        raise NotImplementedError('Must be implemented by derived'
                + ' classes.')

    def sample(self, x, z=None):
        """Sample z given x."""
        raise NotImplementedError('Must be implemented by derived'
                + ' classes.')
