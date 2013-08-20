"""The posterior of SPGP.

Author:
    Ilias Bilionis

Date:
    3/11/2013
"""


__all__ = ['SPGPPosterior']


import numpy as np


class SPGPPosterior(object):
    """The posterior of SPGP."""

    # The prior
    _prior = None

    # The likelihood
    _likelihood = None

    # The SMC parameter
    _gamma = None

    @property
    def prior(self):
        return self._prior

    @property
    def likelihood(self):
        return self._likelihood

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        assert isinstance(value, float)
        assert value >=  0. and value <= 1.
        self._gamma = value

    def __init__(self, prior, likelihood):
        self._prior = prior
        self._likelihood = likelihood
        self._gamma = 1.

    def eval(self, x):
        self.prior.eval(x)
        self.likelihood.eval(x)
        x['log_p'] = self.gamma * x['log_like'] + x['log_prior']

    def d_eval(self, x):
        self.prior.d_eval(x)
        self.likelihood.d_eval(x)
        x['log_p'] = self.gamma * x['log_like'] + x['log_prior']
        x['d_log_p_xb'] = self.gamma * x['d_log_like_xb'] + x['d_log_prior_xb']
        x['d_log_p_log_b'] = self.gamma * x['d_log_like_log_b'] + x['d_log_prior_log_b']
        x['d_log_p_log_c'] = self.gamma * x['d_log_like_log_c'] + x['d_log_prior_log_c']
        x['d_log_p_log_sig'] = self.gamma * x['d_log_like_log_sig'] + x['d_log_prior_log_sig']

    def update(self, x):
        """Call this if gamma changed."""
        x['log_p'] = self.gamma * x['log_like'] + x['log_prior']
        x['d_log_p_xb'] = self.gamma * x['d_log_like_xb'] + x['d_log_prior_xb']
        x['d_log_p_log_b'] = self.gamma * x['d_log_like_log_b'] + x['d_log_prior_log_b']
        x['d_log_p_log_c'] = self.gamma * x['d_log_like_log_c'] + x['d_log_prior_log_c']
        x['d_log_p_log_sig'] = self.gamma * x['d_log_like_log_sig'] + x['d_log_prior_log_sig']