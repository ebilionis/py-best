"""The base class of all solvers.

Author:
    Ilias Bilionis

Date:
    12/2/2012

"""


__all__ = ['Solver']


class Solver(object):
    """The base class of all solvers."""

    # The dimension of each input
    _k_of = None

    # The number of samples of the fixed input
    _n_of = None

    # The number of input components
    _s = None

    # The number of outputs
    _q = None

    # The fixed inputs
    _X_fixed = None

    @property
    def k_of(self):
        """Return the dimensions of each input."""
        return self._k_of

    @property
    def s(self):
        """Get the number of input components."""
        return self._s

    @property
    def q(self):
        """Get the number of outputs."""
        return self._q

    @property
    def X_fixed(self):
        """Get the fixed inputs.

        X_fixed must be a list of dimensions self.s - 1. Each component of the
        list must be a numpy array with dimensions self.n_of_fixed x self.k_of[1:]
        """
        return self._X_fixed

    @property
    def n_of_fixed(self):
        """The number of samples of the fixed input."""
        return self._n_of_fixed

    def __init__(self, s=1, q=1):
        """Initialize the object."""
        self._s = s
        self._q = q
        self._n_of_fixed = [0] * (self.s - 1)
        self._X_fixed = [None] * (self.s - 1)
        self._k_of = [0] * self.s

    def __call__(self, xi, Y=None):
        """Evaluate the solver.

        Arguments:
        xi      ---     The random input variables.

        Keyword Arguments:
        Y       ---     The output vector. If not specified, then it should be
                        allocated and returned.

        """
        raise NotImplementedError('Deriving classes must override this one.')
