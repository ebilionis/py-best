"""A SPGP surrogate.

Author:
    Ilias Bilionis

Date:
    3/13/2013
"""


__all__ = ['SPGPSurrogate']


import numpy as np
from ._spgp import *
from ..maps import Function


class SPGPSurrogate(Function):
    """A SPGP surrogate for one output."""

    # Observed inputs
    _X = None

    # Observed outputs
    _Y = None

    # Pseudo-inputs
    _xb = None

    # Hyper-parameters
    _hyp = None

    # The constant mean
    _y_mean = None

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y

    @property
    def xb(self):
        return self._xb

    @property
    def hyp(self):
        return self._hyp

    @property
    def y_mean(self):
        return self._y_mean

    def __init__(self, X, Y, xb, hyp, y_mean=None):
        super(SPGPSurrogate, self).__init__(X.shape[1], Y.shape[1],
                                            name='SPGP Surrogate')
        if y_mean is None:
            self._y_mean = np.zeros(self.Y)
        else:
            self._y_mean = y_mean
        self._X = X
        self._Y = Y
        self._xb = xb
        self._hyp = hyp

    def __call__(self, x, return_variance=False, add_noise=False):
        mu = np.zeros((x.shape[0], self.num_output))
        s2 = np.zeros((x.shape[0], self.num_output))
        for i in range(self.num_output):
            mu[:, i:(i+1)], s2[:, i:(i+1)] = spgp_pred(self.Y[:, i:(i+1)], self.X, self.xb, x, self.hyp)
        mu += self.y_mean
        if add_noise:
            s2 += np.exp(self.hyp[-1]/2)
        if return_variance:
            return mu, s2
        else:
            return mu

    def sample(self, x, num_samples=1, add_noise=False):
        """Sample from the surrogate."""
        mu, s2 = self(x, return_variance=True, add_noise=add_noise)
        samples = []
        for i in range(num_samples):
            samples.append(mu + np.sqrt(s2) * np.random.randn(*mu.shape))
        return samples