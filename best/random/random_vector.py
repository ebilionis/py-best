"""A class representing a random vector.

Author:
    Ilias Bilionis

Date:
    8/11/2013
"""

import numpy as np
import math
import itertools
import scipy.stats as stats
import best


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
        return self.support.num_dim

    @property
    def name(self):
        return self._name

    def __init__(self, support, name='Random Vector'):
        """Initialize the object.

        Arguments:
            num_dim     ---     The number of dimensions of the random
                                vector. It must be a best.Domain
            support     ---     The support of the random vector. If
                                None, then it is assume to be all space.

        Keyword Arguments:
            name        ---     A name for the random vector.
        """
        if not isinstance(support, best.Domain):
            support = best.DomainRectangle(support)
        self._support = support
        assert isinstance(name, str)
        self._name = name

    def __str__(self):
        """Return a string representation of the object."""
        s = 'Random Vector: ' + self.name + '\n'
        s += str(self.support)
        return s

    def rvs(self, num_samples=1):
        """Return a sample of the random vector."""
        raise NotImplementedError()

    def pdf(self, x):
        """Return the value of the pdf of the random vector at x."""
        return np.exp(self.logpdf(x))

    def cdf(self, x):
        """Return the value of the cdf of the random vector at x."""
        return np.exp(self.logcdf(x))

    def sf(self, x):
        """Survival function."""
        return np.exp(self.logsf(x))

    def logsf(self, x):
        """Log of the survival function."""
        raise NotImplementedError()

    def logpdf(self, x):
        """Return the log of the pdf."""
        raise NotImplementedError()

    def logcdf(self, x):
        """Return the log of the cdf."""
        raise NotImplementedError()

    def moment(self, n):
        """Non-central n-th moment."""
        raise NotImplementedError()

    def entropy(self):
        """The entropy of the distribution."""
        raise NotImplementedError()

    def mean(self):
        """Return the mean of the distribution."""
        raise NotImplementedError()

    def var(self):
        """Return the variance of the distribution."""
        raise NotImplementedError()

    def std(self):
        """Return the standard deviation of the distribution."""
        raise NotImplementedError()


class RandomVectorIndependent(RandomVector):

    """A class representing a random vector with independent components."""

    # A list of random variables
    _component = None

    @property
    def component(self):
        return self._component

    def __getitem__(self, i):
        """Get one of the random variables."""
        return self.component[i]

    def __init__(self, components, name='Independent Random Vector'):
        """Initialize the object.

        Arguments:
            components   ---     A container of random variables.

        Keyword Arguments
            name    ---     A name for the random vector.
        """
        assert (isinstance(components, list)
                or isinstance(components, tuple))
        box = []
        for rv in components:
            box.append(rv.interval(1))
        self._component = components
        support = best.DomainRectangle(box)
        super(RandomVectorIndependent, self).__init__(support, name=name)

    def __str__(self):
        """Return a string representation of the object."""
        s = super(RandomVectorIndependent, self).__str__() + '\n'
        return s

    def rvs(self, num_samples=1):
        """Take random samples."""
        x = []
        for rv in self.component:
            x.append(rv.rvs(num_samples))
        return np.vstack(x).T

    def logpdf(self, x):
        """Return the logarithm of the pdf."""
        x = np.atleast_2d(x)
        tmp = np.ndarray(x.shape)
        for rv, i in itertools.izip(self.component, range(self.num_dim)):
            tmp[:, i] = np.log(rv.pdf(x[:, i]))
        res = tmp.sum(axis=1)
        if res.shape[0] == 1:
            return res[0]
        else:
            return res

    def logcdf(self, x):
        """Return the logarithm of the cdf."""
        x = np.atleast_2d(x)
        tmp = np.ndarray(x.shape)
        for rv, i in itertools.izip(self.component, range(self.num_dim)):
            tmp[:, i] = math.log(rv.cdf(x[:, i]))
        res = tmp.sum(axis=1)
        if res.shape[0] == 1:
            return res[0]
        else:
            return res

    def logsf(self, x):
        """Return the logartihm of the survival function."""
        x = np.atleast_2d(x)
        tmp = np.ndarray(x.shape)
        for rv, i in itertools.izip(self.component, range(self.num_dim)):
            tmp[:, i] = math.log(rv.sf(x[:, i]))
        res = tmp.sum(axis=1)
        if res.shape[0] == 1:
            return res[0]
        else:
            return res


class RandomVariableConditional(object):

    """A random variable conditioned to live in a subinterval."""

    # The original random variable
    _random_variable = None

    # The sub interval (tuple)
    _subinterval = None

    # The log of the probability of the sub interval
    _log_pdf_subinterval = None

    @property
    def random_variable(self):
        return self._random_variable

    @property
    def subinterval(self):
        return self._subinterval

    @property
    def log_pdf_subinterval(self):
        return self._log_pdf_subinterval

    def _compute_log_pdf_subinterval(self):
        """Compute the log of the pdf of the subinterval."""
        return math.log(self.random_variable.cdf(self.subinterval[1])
                        - self.random_variable.cdf(self.subinterval[0]))

    def __init__(self, random_variable, subinterval,
                name='Conditioned Random Variable'):
        """Initialize the object.

        Argument:
            random_variable --- The underlying random variable.
            sub_interval    --- The sub interval.

        Keyword Arguments
            name            --- A name for the random variable.
        """
        self._random_variable = random_variable
        self._subinterval = subinterval
        self._log_pdf_subinterval = self._compute_log_pdf_subinterval()
        assert isinstance(name, str)
        self._name = name

    def interval(self, prob):
        """Return the interval up to probability prob."""
        tmp = self.random_variable.interval(prob)
        res = ()
        if tmp[0] <= self.subinterval[0]:
            res += (self.subinterval[0], )
        else:
            res += (tmp[0], )
        if tmp[1] >= self.subinterval[1]:
            res += (self.subinterval[1], )
        else:
            res += (tmp[1], )
        return res

    def __str__(self):
        """Return a string representation of the object."""
        s = 'Conditioned Random Variable: ' + self.name + '\n'
        s += 'Original interval: ' + str(self.random_variable.interval(1)) + '\n'
        s += 'Subinterval: ' + str(self.subinterval)
        return s

    def pdf(self, x):
        """Return the pdf at x."""
        return np.exp(self.logpdf(x))

    def logpdf(self, x):
        """Return the log of the pdf at x."""
        return self.random_variable.logpdf(x) - self.log_pdf_subinterval

    def cdf(self, x):
        """Return the cdf at x."""
        return np.exp(self.logcdf(x))

    def logcdf(self, x):
        """Return the log of the cdf at x."""
        return self.random_variable.logcdf(x) - self.log_pdf_subinterval


class RandomVectorConditional(RandomVectorIndependent):

    """A random vector of independent variables living in a subdomain."""

    # The underlying random vector
    _random_vector = None

    # The log of the probability of the subdomain
    _log_pdf_subdomain = None

    @property
    def random_vector(self):
        return self._random_vector

    @property
    def log_pdf_subdomain(self):
        return self._log_pdf_subdomain

    def _compute_log_pdf_subdomain(self):
        """Compute the log of the pdf of the subdomain."""
        res = 0.
        for rv in self.component:
            res += rv.log_pdf_subinterval
        return res

    def __init__(self, random_vector, subdomain, name='Name'):
        """Initialize the object.

        Arguments:
            random_vector       ---     The random vector to be conditioned.
            subdomain           ---     The subdomain.

        Keyword Arguments
            name                ---     A name of the random vector.
        """
        if isinstance(random_vector, RandomVectorConditional):
            random_vector = random_vector.random_vector
        assert isinstance(random_vector, RandomVectorIndependent)
        self._random_vector = random_vector
        subdomain = np.array(subdomain)
        # Construct independent components
        comp = []
        for i in range(self.random_vector.num_dim):
            comp.append(RandomVariableConditional(self.random_vector[i],
                                                  subdomain[i, :]))
        super(RandomVectorConditional, self).__init__(comp, name=name)
        self._log_pdf_subdomain = self._compute_log_pdf_subdomain()

    def __str__(self):
        """Return a string representation of the object."""
        s = super(RandomVectorConditional, self).__str__() + '\n'
        s += 'Original:\n'
        s += str(self.random_vector)
        s += 'Prob of subdomain: ' + str(math.exp(self.log_pdf_subdomain))
        return s