"""A class representing a random vector.

Author:
    Ilias Bilionis

Date:
    8/11/2013
"""

import numpy as np
import math
import itertools
import scipy.stats
import best
from random_variable import RandomVariableConditional


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

    def _pdf(self, x):
        """Return the pdf at x.

        Assume that x is already in the domain.
        """
        raise NotImplementedError()

    def pdf(self, x):
        x = np.atleast_2d(x)
        size = x.shape[:-1]
        num_samples = np.prod(size)
        num_dim = x.shape[-1]
        assert num_dim == self.num_dim
        res = np.ndarray(num_samples)
        x = x.reshape((num_samples, num_dim))
        for i in range(num_samples):
            if self.support.is_in(x[i, :]):
                res[i] = self._pdf(x[i, :])
            else:
                res[i] = 0.
        if num_samples == 1:
            return res[0]
        else:
            return res.reshape(size)

    def _rvs(self):
        """Return a sample of the random variable."""
        raise NotImplementedError()

    def rvs(self, size=1):
        """Return a sample of the random vector."""
        if isinstance(size, int):
            size = (size, )
        else:
            size = tuple(size)
        num_samples = np.prod(size)
        x = np.ndarray((num_samples, self.num_dim))
        for i in range(num_samples):
            x[i, :] = self._rvs()
        if num_samples == 1:
            return x[0, :]
        return x.reshape(size + (self.num_dim, ))

    def moment(self, n):
        """Return non-centered n-th moment of the random variable."""
        raise NotImplementedError()

    def mean(self):
        """Return the mean of the random variable."""
        if self._computed_mean is None:
            self._computed_mean = self.moment(1)
        return self._computed_mean

    def var(self):
        """Return the variance of the random variable."""
        if self._computed_variance is None:
            m = self.mean()
            m2 = self.moment(2)
            self._computed_variance = m2 - m ** 2
        return self._computed_variance

    def std(self):
        """Return the standard deviation of the random variable."""
        return np.sqrt(self.var())

    def expect(self, func=None, args=()):
        """Expected value of a function with respect to the distribution.

        Keyword Arguments:
            func        ---     The function.
            args        ---     Arguments to the function.
        """
        raise NotImplementedError()

    def stats(self):
        """Return mean, variance, skewness and kurtosis."""
        m1 = self.moment(1)
        m2 = self.moment(2) - m1 ** 2
        m3 = self.moment(3)
        m4 = self.moment(4) - m2 ** 2
        m = m1
        v = m2
        s = np.sqrt(v)
        sk = (m3 - 3. * m * s **2 - m ** 3) / (s ** 3)
        # 4-th centered moment
        k = m4 / (m2 ** 2)
        return m, v, sk, k


class RandomVectorIndependent(RandomVector):

    """A class representing a random vector with independent components."""

    # A list of random variables
    _component = None

    # The mean (if computed already)
    _computed_mean = None

    # The variance (if computed already)
    _computed_variance = None

    # Probability density in the original domain
    _pdf_domain = None

    @property
    def component(self):
        return self._component

    @property
    def pdf_domain(self):
        return self._pdf_domain

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
        self._pdf_domain = 1.
        for i in range(self.num_dim):
            if isinstance(self.component[i], RandomVariableConditional):
                self._pdf_domain *= self.component[i].pdf_subinterval

    def __str__(self):
        """Return a string representation of the object."""
        s = super(RandomVectorIndependent, self).__str__() + '\n'
        s += 'pdf of domain: ' + str(self.pdf_domain)
        return s

    def _pdf(self, x):
        """Evaluate the pdf at x."""
        return np.prod([rv.pdf(x[i])
                        for rv, i in itertools.izip(self.component,
                                                    range(self.num_dim))])

    def _rvs(self):
        """Take a sample from the random variable."""
        return np.array([rv.rvs() for rv in self.component])

    def moment(self, n):
        """Return the n-th non-centered moment of the random variable."""
        return np.array([rv.moment(n) for rv in self.component])

    def split(self, dim, pt=None):
        """Split the random vector along dimension dim at point pt.

        Arguments:
            dim     ---     The splitting dimension.

        Keyword Arguments:
            pt      ---     The splitting point. If None, then the median
                            of that dimension is used.
        """
        comp1 = list(self.component)
        comp2 = list(self.component)
        tmp_rv = RandomVariableConditional(self.component[dim],
                                           self.component[dim].interval(1.))
        comp1[dim], comp2[dim] = tmp_rv.split(pt=pt)
        return RandomVectorIndependent(comp1), RandomVectorIndependent(comp2)