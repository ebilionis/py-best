"""A proposal for MCMC on SPGP.

Author:
    Ilias Bilionis

Date:
    3/11/2013
"""

import numpy as np
from copy import deepcopy
import math


class SPGPProposal(object):
    """A proposal for MCMC on SPGP."""
    
    # The time steps
    _dt = None
    
    @property
    def dt(self):
        return self._dt
    
    @property
    def dt_xb(self):
        return self._dt[0]
    
    @property
    def dt_log_b(self):
        return self._dt[1]
    
    @property
    def dt_log_c(self):
        return self._dt[2]
    
    @property
    def dt_log_sig(self):
        return self._dt[3]
    
    @dt.setter
    def dt(self, value):
        assert isinstance(value, np.ndarray)
        self._dt = value
    
    def __init__(self, dt=None):
        if dt is not None:
            dt = np.array(dt)
            self._dt = dt
        else:
            self._dt = np.ndarray(4)
            self._dt.fill(0.1)
    
    def propose_xb(self, x_old):
        x_new = deepcopy(x_old)
        x_new['xb'] = (x_old['xb']
                       + 0.5 * self.dt_xb * x_old['d_log_p_xb']
                       + math.sqrt(self.dt_xb) * np.random.randn(*x_old['xb'].shape))
        return x_new
    
    def propose_log_b(self, x_old):
        x_new = deepcopy(x_old)
        x_new['log_b'] = (x_old['log_b']
                          + 0.5 * self.dt_log_b * x_old['d_log_p_log_b']
                          + math.sqrt(self.dt_log_b) * np.random.randn(*x_old['log_b'].shape))
        return x_new
    
    def propose_log_c(self, x_old):
        x_new = deepcopy(x_old)
        x_new['log_c'] = (x_old['log_c']
                          + 0.5 * self.dt_log_c * x_old['d_log_p_log_c']
                          + math.sqrt(self.dt_log_c) * np.random.randn())
        return x_new
    
    def propose_log_sig(self, x_old):
        x_new = deepcopy(x_old)
        x_new['log_sig'] = (x_old['log_sig']
                          + 0.5 * self.dt_log_sig * x_old['d_log_p_log_sig']
                          + math.sqrt(self.dt_log_sig) * np.random.randn())
        return x_new
    
    def eval_xb(self, x_new, x_old):
        """Assuming that the dictionaries are properly filled in."""
        tmp = x_old['xb'] + 0.5 * self.dt_xb * x_old['d_log_p_xb'] - x_new['xb']
        tmp = tmp.flatten(order='F') / math.sqrt(self.dt_xb)
        return -0.5 * np.dot(tmp, tmp)
    
    def eval_log_b(self, x_new, x_old):
        tmp = x_old['log_b'] + 0.5 * self.dt_log_b * x_old['d_log_p_log_b'] - x_new['log_b']
        tmp /= math.sqrt(self.dt_log_b)
        return -0.5 * np.dot(tmp, tmp)
    
    def eval_log_c(self, x_new, x_old):
        tmp = x_old['log_c'] + 0.5 * self.dt_log_c * x_old['d_log_p_log_c'] - x_new['log_c']
        tmp /= math.sqrt(self.dt_log_c)
        return -0.5 * (tmp ** 2)
    
    def eval_log_sig(self, x_new, x_old):
        tmp = x_old['log_sig'] + 0.5 * self.dt_log_sig * x_old['d_log_p_log_sig'] - x_new['log_sig']
        tmp /= math.sqrt(self.dt_log_sig)
        return -0.5 * (tmp ** 2)