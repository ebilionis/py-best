"""Models a distribution of the form:

    p(x, z) = p(z | x) p(x).

Author:
    Ilias Bilionis

Date:
    2/21/2013

"""


__all__ = ['ProductDistribution']


import numpy as np
from . import Distribution
from . import ConditionalDistribution


class ProductDistribution(Distribution):

    # Probability of z conditional on x
    _pzcx = None

    # Probability of x
    _px = None

    @property
    def pzcx(self):
        return self._pzcx

    @property
    def px(self):
        return self._px

    def __init__(self, pzcx, px, name='Product distribution'):
        """Initialize the object.

        Arguments:
            pzcx        ---         p(z|x)
            px          ---         p(x)
        """
        assert isinstance(pzcx, ConditionalDistribution)
        assert isinstance(px, Distribution)
        self._pzcx = pzcx
        self._px = px
        super(ProductDistribution, self).__init__(pzcx.num_input +
                px.num_input, name=name)

    def __call__(self, x):
        """Evaluate the log at x."""
        x0 = x[:self.px.num_input]
        z = x[self.px.num_input:]
        return self.pzcx(z, x0) + self.px(x0)

    def sample(self):
        """Sample the distribution."""
        x = self.px.sample()
        z = self.pzcx.sample(x)
        return np.hstack([x, z])
