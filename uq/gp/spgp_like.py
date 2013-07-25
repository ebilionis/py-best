"""Likelihood for SPGP.

Author:
    Ilias Bilionis
    
Date:
    3/11/2013
"""

import numpy as np
import math
from uq.gp.spgp import *


class SPGPLikelihood(object):
    """SPGP Likelihood."""
    
    # Inputs
    _X = None
    
    # Outputs
    _Y = None
    
    # Number of pseudo inputs
    _num_pseudo = None
    
    # For stability
    _d = None
    
    @property
    def X(self):
        return self._X
    
    @property
    def Y(self):
        return self._Y
    
    @property
    def num_pseudo(self):
        return self._num_pseudo
    
    @property
    def d(self):
        return self._d
    
    def __init__(self, X, Y, num_pseudo, d=1e-6):
        self._X = X
        self._Y = Y
        self._num_pseudo = num_pseudo
        self._d = d

    def eval(self, x):
        """Evaluate the log likelihood."""
        w = np.hstack([x['xb'].flatten(order='F'), x['log_b'], x['log_c'],
                      x['log_sig']])
        lw = 0.
        for i in range(self.Y.shape[1]):
            lw += spgp_lik(w, self.Y[:, i:(i+1)], self.X, self.num_pseudo,
                          d=self.d, compute_der=False)
        x['log_like'] = -lw
    
    def d_eval(self, x):
        """Evaluate the log likelihood and its derivative."""
        dim = self.X.shape[1]
        num_dim_xb = self.num_pseudo * dim
        w = np.hstack([x['xb'].flatten(order='F'), x['log_b'], x['log_c'],
                      x['log_sig']])
        x['log_like'] = 0.
        x['d_log_like_xb'] = np.zeros(x['xb'].shape)
        x['d_log_like_log_b'] = np.zeros(x['log_b'].shape)
        x['d_log_like_log_c'] = 0.
        x['d_log_like_log_sig'] = 0.
        for i in range(self.Y.shape[1]):
            lw = spgp_lik(w, self.Y[:, i:(i+1)], self.X, self.num_pseudo,
                          d=self.d, compute_der=True)
            x['log_like'] += -lw[0]
            d_log_like_xb = -lw[1][:num_dim_xb]
            x['d_log_like_xb'] += d_log_like_xb.reshape(x['xb'].shape, order='F')
            x['d_log_like_log_b'] += -lw[1][num_dim_xb:(num_dim_xb + dim)]
            x['d_log_like_log_c'] += -lw[1][-2]
            x['d_log_like_log_sig'] += -lw[1][-1]