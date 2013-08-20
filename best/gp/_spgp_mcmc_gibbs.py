"""MCMC-Gibbs for SPGP.

Author:
    Ilias Bilionis

Date:
    3/11/2013
"""


__all__ = ['SPGPMCMCGibbs']


import numpy as np
import math
#import matplotlib.pylab as plt


class SPGPMCMCGibbs(object):
    """MCMC-Gibbs for SPGP."""

    # The target distribution
    _target = None

    # The proposal
    _proposal = None

    # Count samples per variable
    _count = None

    # Counted accepted samples per variable
    _accepted = None

    # Number of gibbs steps per variable
    _num_gibbs = None

    # Verbosity
    _verbose = None

    @property
    def target(self):
        return self._target

    @property
    def proposal(self):
        return self._proposal

    @property
    def count(self):
        return self._count

    @property
    def accepted(self):
        return self._accepted

    @property
    def acceptance_rate(self):
        return self._accepted / self._count

    @property
    def num_gibbs(self):
        return self._num_gibbs

    @property
    def verbose(self):
        return self._verbose

    def __init__(self, posterior, proposal, num_gibbs=[1, 1, 1, 1],
                 verbose=False):
        self._num_gibbs = np.array(num_gibbs, dtype='int')
        self._target = posterior
        self._proposal = proposal
        self._count = np.zeros(4, dtype='float')
        self._accepted = np.zeros(4, dtype='float')
        self._verbose = verbose

    def _get_var_idx(self, varname):
        if varname == 'xb':
            return 0
        elif varname == 'log_b':
            return 1
        elif varname == 'log_c':
            return 2
        elif varname == 'log_sig':
            return 3

    def sample(self, x, steps=1):
        for i in range(steps):
            if self.verbose:
                print i, 'MCMC step:'
            self._sample_gibbs(x, 'xb')
            self._sample_gibbs(x, 'log_b')
            self._sample_gibbs(x, 'log_c')
            self._sample_gibbs(x, 'log_sig')
            #print 'log_p: ', x['log_p'], np.exp(-x['log_b']/2), np.exp(x['log_c']/2), np.exp(x['log_sig']/2)
            #print self.acceptance_rate
            #plt.clf()
            #plt.plot(x['xb'], np.zeros(x['xb'].shape), 'ro', markersize=10)
            #plt.plot(np.exp(-x['log_b']/2)[0], math.exp(x['log_c']/2), 'r.')
            #plt.pause(1e-1)

    def _sample_gibbs(self, x, var_name):
        if self.verbose:
            print ' Sampling', var_name
        # Get idx of variable
        var_idx = self._get_var_idx(var_name)
        prop_func = 'self.proposal.propose_' + var_name + '(x)'
        prop_eval = 'self.proposal.eval_' + var_name + '('
        prop_eval_old_new = prop_eval + 'x, x_new)'
        prop_eval_new_old = prop_eval + 'x_new, x)'
        for i in range(self.num_gibbs[var_idx]):
            self._sample_single_gibbs(x, var_idx, prop_func, prop_eval_old_new,
                                      prop_eval_new_old)

    def _sample_single_gibbs(self, x, var_idx, prop_func,
                            prop_eval_old_new, prop_eval_new_old):
        # Propose a move
        x_new = eval(prop_func)
        # Evaluate the posterior at the new point
        self.target.d_eval(x_new)
        # Compute a1 = p(x_new) / p(x)
        log_a1 = x_new['log_p'] - x['log_p']
        # Compute a2 = p(x | x_new) / p(x_new | x)
        log_a2 = eval(prop_eval_old_new) - eval(prop_eval_new_old)
        # Compute a = a1 * a2
        log_a = log_a1 + log_a2
        if self.verbose:
            print '  log_a1 =', log_a1
            print '  log_a2 =', log_a2
            print '  log_a =', log_a
        # Accept if a >= 1
        if log_a >= 0. or np.random.rand() < math.exp(log_a):
            x.update(x_new)
            self._accepted[var_idx] += 1
        # Increase sample counter
        self._count[var_idx] += 1