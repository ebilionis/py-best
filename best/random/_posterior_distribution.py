"""A class representing a posterior distribution.

Author:
    Ilias Bilionis

Date:
    1/15/2013

"""

__all__ = ['PosteriorDistribution']


from ..maps import Function
from . import LikelihoodFunction
from . import Distribution


class PosteriorDistribution(LikelihoodFunction):
    """A class representing a posterior distribution.

    The likelihood function.
    The class requires a likelihood object which can be any class implementing:
    + likelihood.__call__(x):   Giving the log likelihood at a particular x.
    Notice that x here is the parameter of the likelihood not the data.
    It is the responsibility of the user to make sure that the likelihood function,
    correctly captures the dependence on the data.

    The prior distribution.
    The prior distribution is any object which implements:
    + prior.__call__(x):    Giving the log of the prior.

    Overall, this class represents the following function:
        p(x | y, \gamma) \propto p(y | x)^\gamma p(x).

    Again, I mention that specifying 'y' is the responsibility of the user.
    It is not directly needed in this class. All we use is p(y | x) as a
    function x only, y being fixed and implied.
    The parameter gamma plays the role of a regularizing parameter.
    The default value is 1. We have explicitely included it, because the main
    purpose of this class is to be used within the Sequential Monte Carlo framework.
    """

    # The likelihood function
    _likehood = None

    # The prior distribution
    _prior = None

    # The regularizing parameter
    _gamma = None

    @property
    def likelihood(self):
        """Get the likelihood function."""
        return self._likelihood

    @likelihood.setter
    def likelihood(self, value):
        """Set the likelihood function."""
        assert value is not None, 'The likelihood must be specified.'
        assert isinstance(value, LikelihoodFunction)
        self._likelihood = value

    @property
    def prior(self):
        """Get the prior distribution."""
        return self._prior

    @prior.setter
    def prior(self, value):
        """Set the prior distribution."""
        assert value is not None, 'The prior must be specified.'
        assert isinstance(value, Distribution)
        self._prior = value

    @property
    def gamma(self):
        """Get the regularizing parameter."""
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        """Set the regularizing parameter."""
        if not isinstance(value, float):
            raise TypeError('Gamma must be a float.')
        if value < 0. or value > 1.:
            raise ValueError('Gamma must be in [0, 1]')
        self._gamma = value

    def __init__(self, likelihood=None, prior=None, gamma=1.,
            name='Posterior Distribution'):
        """Initialize the object.

        Keyword Arguments:
        likelihood  ---     The likelihood function.
        prior       ---     The prior distribution.
        gamma       ---     The regularizing parameter.
        name        ---     A name for the function.
        """
        self.likelihood = likelihood
        self.prior = prior
        assert self.likelihood.num_input == self.prior.num_input
        super(PosteriorDistribution, self).__init__(self.likelihood.num_input,
                name=name)
        self.gamma = gamma

    def __call__(self, x, report_all=False):
        """Evaluate the log of the pdf at x.

        Arguments:
            x   ---     The point of evalutation.

        Keyword Arguments:
            report_all  ---     If set to True, then it returns
                                a dictionary of all the values used
                                to compose the log of the posterior (see
                                below for details). Otherwise, it simply
                                returns the log of the posterior.

        Details on the dictionary returned.
        ---------------------------------------------------
        The function returns a dictionary r that contains:
        + r['log_p']:       The log of the pdf at x.
        + r['log_like']:    The log of the likelihood at x.
        + r['log_prior']:   The log of the prior at x.
        + r['gamma']:       The current gamma.
        """
        log_prior = self.prior(x)
        if log_prior < -1e90:
            log_like = log_prior
        else:
            log_like = self.likelihood(x)
        log_p = self.gamma * log_like + log_prior
        if report_all:
            r = {}
            r['log_p'] = log_p
            r['log_like'] = log_like
            r['log_prior'] = log_prior
            r['gamma'] = self.gamma
            return r
        else:
            return log_p

    def _to_string(self, pad):
        """Return a string representation of the object."""
        s = super(PosteriorDistribution, self)._to_str(pad) + '\n'
        s += pad + ' Likelihood:\n'
        s += self.likelihood._to_string(pad + ' ') + '\n'
        s += pad + ' Prior:\n'
        s += self.prior._to_string(pad + ' ')
        return s
