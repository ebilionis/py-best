"""The base class of all likelihood functions.

Author:
    Ilias Bilionis

Date:
    1/21/2013

"""


__all__ = ['LikelihoodFunction']


from ..maps import Function


class LikelihoodFunction(Function):
    """The base class of all likelihood functions.

    A likelihood function is a actually a function of the hyper-parameters
    (or simply the parameters) of the model and the data. In Bayesian statistics,
    it basicaly models:
        l(x) = log(p(D |x ).    (1)
    D, is the data and it should be set either at the constructor or with
        likelihood.data = data.
    The log of the likelihood at x for a given D (see Eq. (1)) is evaluated by:
        likelihood.__call__(x),
    which is a function that should be implemented by the user.
    """

    # The observed data
    _data = None

    @property
    def data(self):
        """Get the observed data."""
        return self._data

    @data.setter
    def data(self, value):
        """Set the observed data."""
        self._data = value

    def __init__(self, num_input, data=None, name='Likelihood function',
                 log_l_wrapped=None):
        """Initialize the object.

        Arguments:
            num_input   ---     The number of inputs (i.e. number of parameters
                                of the likelihood function.)

        Keyword Arguments:
            data            ---     The observed data.
            name            ---     The name of the likelihood function.
            log_l_wrapped   ---     A normal function that implements, the
                                    likelihood.
        """
        super(LikelihoodFunction, self).__init__(num_input, 1, name=name)
        if data is not None:
            self.data = data