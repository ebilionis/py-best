"""A class representing a random variable.

Author:
    Ilias Bilionis

Date:
    8/13/2013
"""


__all__ = ['RandomVariableConditional']


import scipy.stats


class RandomVariableConditional(scipy.stats.rv_continuous):

    """A random variable conditioned to live in a subinterval."""

    # The original random variable
    _random_variable = None

    # The log of the probability of the sub interval
    _pdf_subinterval = None

    @property
    def random_variable(self):
        return self._random_variable

    @property
    def subinterval(self):
        return (self.a, self.b)

    @property
    def pdf_subinterval(self):
        return self._pdf_subinterval

    def _compute_pdf_subinterval(self):
        """Compute the log of the pdf of the subinterval."""
        return (self.random_variable.cdf(self.b) -
                self.random_variable.cdf(self.a))

    def __init__(self, random_variable, subinterval,
                name='Conditioned Random Variable'):
        """Initialize the object.

        Argument:
            random_variable --- The underlying random variable.
            sub_interval    --- The sub interval.

        Keyword Arguments
            name            --- A name for the random variable.
        """
        if isinstance(random_variable, RandomVariableConditional):
            random_variable = random_variable.random_variable
        self._random_variable = random_variable
        super(RandomVariableConditional, self).__init__(a=subinterval[0],
                                                        b=subinterval[1],
                                                        name=name)
        self._pdf_subinterval = self._compute_pdf_subinterval()

    def __str__(self):
        """Return a string representation of the object."""
        s = 'Conditioned Random Variable: ' + self.name + '\n'
        s += 'Original interval: ' + str((self.a, self.b)) + '\n'
        s += 'Subinterval: ' + str(self.subinterval) + '\n'
        s += 'Prob of sub: ' + str(self.pdf_subinterval)
        return s

    def _pdf(self, x):
        """Return the pdf at x."""
        return self.random_variable.pdf(x) / self.pdf_subinterval

    def _cdf(self, x):
        """Return the cdf at x."""
        return ((self.random_variable.cdf(x) -
                 self.random_variable.cdf(self.a)) /
                self.pdf_subinterval)

    def split(self, pt=None):
        """Split the distribution at pt.

        Keyword Arguments:
            pt   ---     The splitting point. If not specified, then the
                        median is used.
        """
        if pt is None:
            pt = self.median()
        sub1 = (self.a, pt)
        sub2 = (pt, self.b)
        rv1 = RandomVariableConditional(self, sub1)
        rv2 = RandomVariableConditional(self, sub2)
        return rv1, rv2