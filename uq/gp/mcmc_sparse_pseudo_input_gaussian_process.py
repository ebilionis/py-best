"""MCMC for SPGP single output.

Author:
    Ilias Bilionis

Date:
    3/11/2013

"""


import numpy as np
import math
from uq.gp.spgp import *


class SPGPPrior(object):
    """A simple prior distribution for the hyper-parameters."""

    # The observed input points
    _X = None

    _num_pseudo = None

    _r_scale = None

    _s_scale = None

    _g_scale = None

    @property
    def X(self):
        return self._X

    @property
    def num_pseudo(self):
        return self._num_pseudo

    @property
    def dim(self):
        return self._X.shape[1]

    @property
    def r_scale(self):
        return self._r_scale

    @property
    def s_scale(self):
        return self._s_scale

    @property
    def g_scale(self):
        return self._g_scale

    def __init__(self, num_pseudo, X, r_scale=0.1, s_scale=1.,
            g_scale=0.001):
        self._num_pseudo = num_pseudo
        self._r_scale = r_scale
        self._s_scale = s_scale
        self._g_scale = g_scale
        self._X = X
        # inputs: pseudo-inputs + lengthscales + signal + noise
        num_input = num_pseudo * X.shape[1] + X.shape[1] + 2

    def sample_pseudo_inputs(self):
        I = np.argsort(np.random.rand(self.X.shape[0]))
        I = I[:self.num_pseudo]
        return self.X[I, :]

    def sample_log_r(self):
        r = np.random.exponential(self.r_scale, (self.dim, ))
        return np.log(r)

    def sample_log_s(self):
        s = np.random.exponential(self.s_scale)
        return math.log(s)

    def sample_log_g(self):
        g = np.random.exponential(self.g_scale)
        return math.log(g)

    def sample(self):
        x = {}
        x['xb'] = self.sample_pseudo_inputs()
        x['log_r'] = self.sample_log_r()
        x['log_s'] = self.sample_log_s()
        x['log_g'] = self.sample_log_g()
        return x

    def eval_xb(self, x, eval_deriv=False):
        x['log_p_xb'] = 0.

    def eval_d_xb(self, x):
        xb = x['xb']
        x['log_p_xb'] = 0.
        x['d_log_p_xb'] = np.zeros(xb.shape)

    def eval_log_r(self, x, eval_deriv=False):
        log_r = x['log_r']
        r = np.exp(log_r)
        x['log_p_log_r'] = np.sum(log_r - self.r_scale * r)

    def eval_d_log_r(self, x):
        log_r = x['log_r']
        r = np.exp(log_r)
        x['log_p_log_r'] = np.sum(log_r - self.r_scale * r)
        x['d_log_p_log_r'] np.ones(log_r.shape) - self.r_scale * r

    def eval_log_s(self, x, eval_deriv=False):
        log_s = x['log_s']
        s = math.exp(log_s)
        x['log_p_log_s'] = log_s - self.s_scale * s

    def eval_d_log_s(self, x):
        log_s = x['log_s']
        s = math.exp(log_s)
        x['log_p_log_s'] = log_s - self.s_scale * s
        x['d_log_p_log_s'] = 1. - self.s_scale * s

    def eval_log_g(self, x, eval_deriv=False):
        log_g = x['log_g']
        g = math.exp(log_g)
        x['log_p_log_g'] = log_g - self.g_scale * g

    def eval_d_log_g(self, x):
        log_g = x['log_g']
        g = math.exp(log_g)
        x['log_p_log_g'] = log_g - self.g_scale * g
        x['d_log_p_log_g'] = 1. - self.g_scale * g

    def eval(self, x):
        self.eval_xb(x)
        self.eval_log_r(x)
        self.eval_log_s(x)
        self.eval_log_g(x)

    def eval_d(self, x):
        self.eval_d_xb(x)
        self.eval_d_log_r(x)
        self.eval_d_log_s(x)
        self.eval_d_log_g(x)

def SPGPLikelihood(object):
    """SPGP Likelihood."""

    # Inputs
    _X = None

    # Outputs
    _Y = None

    # Control numerical stability
    _d = None

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y

    @property
    def d(self):
        return self._d

    def __init__(self, X, Y, d=1e-6):
        self._X = X
        self._Y = Y
        self._d = d

    def eval(self, x):
        xb = x['xb']
        log_b = -2. * x['log_r']
        log_c = 2. * x['log_s']
        log_g = 2. * x['log_g']
        w = np.hstack([xb.flatten(order='F'),
            log_b, log_c, log_g])
        lw = spgp_lik(w, Y, X, self.num_pseudo, d=self.d, compute_der=False)
