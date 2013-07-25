"""A prior for SPGP.

Author:
    Ilias Bilionis

Date:
    3/11/2013
"""

import numpy as np
import math
from uq.gp.spgp import *


class SPGPPrior(object):
    """A prior model for SGPG."""
    
    # Input points
    _X = None
    
    # Number of pseudo inputs
    _num_pseudo = None
    
    # Scale for r
    _r_scale = None
    
    # Scale for s
    _s_scale = None
    
    # Scale for g
    _g_scale = None
    
    @property
    def X(self):
        return self._X
    
    @property
    def num_pseudo(self):
        return self._num_pseudo
    
    @property
    def r_scale(self):
        return self._r_scale
    
    @property
    def g_scale(self):
        return self._g_scale
    
    @property
    def s_scale(self):
        return self._s_scale
    
    def __init__(self, X, num_pseudo, r_scale=0.1, s_scale=1., g_scale=1.):
        self._X = X
        self._num_pseudo = num_pseudo
        self._r_scale = r_scale
        self._s_scale = s_scale
        self._g_scale = g_scale
    
    def sample_xb(self, x):
        I = np.argsort(np.random.randn(self.X.shape[0]))
        I = I[:self.num_pseudo]
        x['xb'] = self.X[I, :]
    
    def sample_log_b(self, x):
        r = np.random.exponential(self.r_scale, (self.X.shape[1], ))
        x['log_b'] = -2. * np.log(r)
    
    def sample_log_c(self, x):
        s = np.random.exponential(self.s_scale)
        x['log_c'] = 2. * math.log(s)
    
    def sample_log_sig(self, x):
        g = np.random.exponential(self.g_scale)
        x['log_sig'] = 2. * math.log(g)
    
    def sample(self):
        """Sample from the prior."""
        x = {}
        x['xb'] = None
        self.sample_xb(x)
        x['log_b'] = None
        self.sample_log_b(x)
        x['log_c'] = None
        self.sample_log_c(x)
        x['log_sig'] = None
        self.sample_log_sig(x)
        return x
    
    def eval_xb(self, x):
        x['log_prior_xb'] = 0.
    
    def d_eval_xb(self, x):
        x['log_prior_xb'] = 0.
        x['d_log_prior_xb'] = np.zeros(x['xb'].shape)
    
    def eval_log_b(self, x):
        log_b = x['log_b']
        r = np.exp(-0.5 * log_b)
        x['log_prior_log_b'] = -0.5 * log_b - r / self.r_scale
    
    def d_eval_log_b(self, x):
        log_b = x['log_b']
        r = np.exp(-0.5 * log_b)
        x['log_prior_log_b'] = (-0.5 * log_b - r / self.r_scale).sum()
        x['d_log_prior_log_b'] = -0.5 + 0.5 * r / self.r_scale
    
    def eval_log_c(self, x):
        log_c = x['log_c']
        s = math.exp(0.5 * log_c)
        x['log_prior_log_c'] = 0.5 * log_c - s / self.s_scale
    
    def d_eval_log_c(self, x):
        log_c = x['log_c']
        s = math.exp(0.5 * log_c)
        x['log_prior_log_c'] = 0.5 * log_c - s / self.s_scale
        x['d_log_prior_log_c'] = 0.5 - 0.5 * s / self.s_scale
    
    def eval_log_sig(self, x):
        log_sig = x['log_sig']
        g = math.exp(0.5 * log_sig)
        x['log_prior_log_sig'] = 0.5 * log_sig - g / self.g_scale
    
    def d_eval_log_sig(self, x):
        log_sig = x['log_sig']
        g = math.exp(0.5 * log_sig)
        x['log_prior_log_sig'] = 0.5 * log_sig - g / self.g_scale
        x['d_log_prior_log_sig'] = 0.5 - 0.5 * g / self.g_scale
    
    def _sum_log_prior(self, x):
        x['log_prior'] = (x['log_prior_xb'] + x['log_prior_log_b']
                          + x['log_prior_log_c'] + x['log_prior_log_sig'])
    
    def eval(self, x):
        """Evaluate the log prior at x."""
        self.eval_xb(x)
        self.eval_log_b(x)
        self.eval_log_c(x)
        self.eval_log_sig(x)
        self._sum_log_prior(x)

    def d_eval(self, x):
        """Evaluate the log prior at x."""
        self.d_eval_xb(x)
        self.d_eval_log_b(x)
        self.d_eval_log_c(x)
        self.d_eval_log_sig(x)
        self._sum_log_prior(x)