"""Relevance Vector Machines Module

Author:
    Ilias Bilionis

Date:
    8/15/2013
"""


import numpy as np
import scipy.linalg
import scipy.optimize
import math
import best.linalg
import best.maps


# Some Actions
REESTIMATE = 0
ADD = 1
DELETE = 2
UNKNOWN = 3


class RelevanceVectorMachine(object):

    """The relevance vector machine class."""

    # The design matrix
    _PHI = None

    # The output data
    _Y = None

    # A scaled version of the design matrix
    _PHIs = None

    # The scales of the design matrix
    _PHI_scales = None

    # The hyper-parameters of the weights
    _alpha = None

    # The precision
    _beta = None

    # The relevant vectors
    _relevant = None

    # The used vectors
    _used = None

    # The relevant weights
    _weights = None

    # The square root of the inverse covariance matrix
    _sigma_sqrt = None

    # The statistics
    _S = None
    _H = None
    _s = None
    _h = None
    _theta = None

    # The log likelihood
    _log_like = None

    # The GSVD
    _gsvd = None

    @property
    def PHI(self):
        return self._PHI

    @property
    def Y(self):
        return self._Y

    @property
    def num_samples(self):
        return self.Y.shape[0]

    @property
    def num_basis(self):
        return self.PHI.shape[1]

    @property
    def num_output(self):
        return self.Y.shape[1]

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta

    @property
    def relevant(self):
        return self._relevant

    @property
    def num_relevant(self):
        return self._relevant.shape[0]

    @property
    def used(self):
        return self._used

    @property
    def PHIs(self):
        return self._PHIs

    @property
    def PHI_scales(self):
        return self._PHI_scales

    @property
    def weights(self):
        return self._weights

    @property
    def sigma_sqrt(self):
        return self._sigma_sqrt

    @property
    def S(self):
        return self._S

    @property
    def H(self):
        return self._H

    @property
    def s(self):
        return self._s

    @property
    def h(self):
        return self._h

    @property
    def theta(self):
        return self._theta

    @property
    def log_like(self):
        return self._log_like

    @property
    def gsvd(self):
        return self._gsvd

    def _to_2d(self, x):
        """Turn an 1D array to a valid 2D array."""
        x = np.atleast_2d(x)
        if x.shape[0] == 1:
            return x.T
        return x

    def _allocate_memory(self):
        """Allocates memory for all data."""
        self._PHIs = np.ndarray(self.PHI.shape)
        self._PHI_scales = np.ndarray(self.num_basis)
        self._S = np.ndarray(self.num_basis)
        self._H = np.ndarray((self.num_basis, self.num_output))
        self._s = np.ndarray(self.num_basis)
        self._h = np.ndarray((self.num_basis, self.num_output))
        self._theta = np.ndarray(self.num_basis)

    def _scale_PHI(self):
        """Scale the design matrix."""
        self._PHI_scales = np.sqrt(np.einsum('ij, ij->j', self.PHI,
                                             self.PHI))
        self._PHIs = np.einsum('ij, j->ij', self.PHI,
                               1. / self.PHI_scales)

    def _fix_used(self):
        """Fix the used vector.

        Assert that relevant is already set.
        """
        self._used = np.ndarray(self.num_basis, dtype='i')
        self._used[:] = -1
        self._used[self.relevant] = np.arange(self.num_relevant,
                                              dtype='i')

    def _fix_relevant_gsvd(self):
        """Fix everything that is required for gsv."""
        # Fix A
        A = self.PHIs[:, self.relevant]
        # Fix B
        B = np.zeros((self.num_relevant, self.num_relevant))
        np.fill_diagonal(B, np.sqrt(self.alpha / self.beta))
        # Compute the gsvd
        self._gsvd = best.linalg.GeneralizedSVD(A, B) # Do not do V
        self._R_lu = scipy.linalg.lu_factor(self.gsvd.R)

    def _finalize(self):
        """Finalize the algorithm."""
        self._weights = np.einsum('ij, i->ij', self.weights,
                                  1. / self.PHI_scales[self.relevant])
        self._sigma_sqrt = np.einsum('ij, i->ij', self.sigma_sqrt,
                                     1. / self.PHI_scales[self.relevant])

    def _compute_log_det_RRT(self):
        return 2. * np.log(np.fabs(np.diag(self._R_lu[0]))).sum()

    def _compute_weights(self, tmpm_mr_q):
        tmpm_mr_q = scipy.linalg.lu_solve(self._R_lu, tmpm_mr_q)

        self._weights = np.dot(self.gsvd.Q, tmpm_mr_q)

        self._sigma_sqrt = (scipy.linalg.lu_solve(self._R_lu, self.gsvd.Q.T,
                                                 trans=1) /
                            math.sqrt(self.beta))

    def _compute_statistics(self):
        """Compute the statistics at the current alphas and beta."""
        tmpm_n_m = np.dot(self.gsvd.U.T, self.PHIs)
        tmpm_mr_m = np.dot(self.gsvd.D1.T, tmpm_n_m)

        tmpm_n_q = np.dot(self.gsvd.U.T, self.Y)
        tmpm_mr_q = np.dot(self.gsvd.D1.T, tmpm_n_q)

        self._S[:] = 1. - np.einsum('ij, ij->j', tmpm_mr_m, tmpm_mr_m)
        self._S *= self.beta

        self._H[:] = np.dot(self.PHIs.T, self.Y)
        self._H -= np.dot(tmpm_mr_m.T, tmpm_mr_q)
        self._H *= self.beta

        self.s[:] = self.S
        self.s[self.relevant] = ((self.alpha * self.S[self.relevant]) /
                                 (self.alpha - self.S[self.relevant]))

        self.h[:] = self.H
        self.h[self.relevant, :] = np.einsum('i, ij, i->ij',
                        self.alpha,
                        self.H[self.relevant, :],
                        1. / (self.alpha - self.S[self.relevant]))

        self.theta[:] = (np.einsum('ij, ij->i', self.h, self.h) / self.num_output
                         - self.s)

        # Compute the likelihood
        # Logarithm of |C|
        log_sum_alpha = np.sum(np.log(self.alpha))
        log_det_RRT = self._compute_log_det_RRT()
        log_det_C = (-log_sum_alpha + log_det_RRT
                     + (self.num_relevant - self.num_samples)
                     * self.beta)

        mean_yCiy = ((np.sum(self.Y ** 2) - np.sum(tmpm_mr_q ** 2)) *
                     (self.beta / self.num_output))

        self._log_like = (-0.5 * math.log(2. * math.pi)
                    -0.5 * (log_det_C + mean_yCiy ) / self.num_samples)

        self._compute_weights(tmpm_mr_q)

    def _compute_delta_log_like(self, i, alpha):
        """Computes the change in log likelihood."""
        return (0.5*(math.log(alpha / math.fabs(alpha + self.s[i]))
                + (self.s[i] + self.theta[i]) / (alpha + self.s[i]))
                / self.num_samples)

    def _step(self):
        """Perform a signle step of the algorithm.

        Return the maximum change in log likelihood.
        """
        self._compute_statistics()

        # Find the best action
        max_action = UNKNOWN
        max_delta_log_like = 0.
        max_i = - 1
        max_alpha = 0.
        for i in xrange(self.num_basis):
            if self.theta[i] > 0. and self.used[i] >= 0:
                # REESTIMATE
                alpha_new_i = self.s[i] ** 2 / self.theta[i]
                #print '***', self.used[i], i, self.relevant
                delta_log_like = (self._compute_delta_log_like(i, alpha_new_i)
                                  - self._compute_delta_log_like(i, self.alpha[self.used[i]]))
                if delta_log_like > max_delta_log_like:
                    max_delta_log_like = delta_log_like
                    max_alpha = alpha_new_i
                    max_i = i
                    max_action = REESTIMATE
                #print 'RES, ', i, delta_log_like
            elif (self.theta[i] > 0. and self.used[i] < 0
                  and self.num_relevant < self.num_samples):
                # ADD
                alpha_new_i = self.s[i] ** 2 / self.theta[i]
                delta_log_like = self._compute_delta_log_like(i, alpha_new_i)
                if delta_log_like > max_delta_log_like:
                    max_delta_log_like = delta_log_like
                    max_alpha = alpha_new_i
                    max_i = i
                    max_action = ADD
                #print 'ADD, ', i, delta_log_like
            elif (self.theta[i] < 0. and self.used[i] >= 0
                  and self.num_relevant > 1 and False):
                # DELETE
                delta_log_like = - self._compute_delta_log_like(i, self.alpha[self.used[i]])
                if delta_log_like > max_delta_log_like:
                    max_delta_log_like = delta_log_like
                    max_alpha = float('inf')
                    max_i = i
                    max_action = DELETE
                #print 'DEL, ', i, delta_log_like


        # Perform the best action
        if max_action == REESTIMATE:
            self.alpha[self.used[max_i]] = max_alpha
        elif max_action == ADD:
            self.used[max_i] = self.num_relevant
            self.relevant.resize(self.num_relevant + 1)
            self.relevant[-1] = max_i
            self._alpha = np.hstack([self.alpha, max_alpha])
        elif max_action == DELETE:
            #print 'To remove: ', max_i, self.used[max_i], self.alpha[self.used[max_i]]
            r = self.used[max_i]
            mr = self.num_relevant
            tmp = np.ndarray(mr - 1, dtype='i')
            tmp[:r] = self.relevant[:r]
            tmp[r:] = self.relevant[r + 1:]
            self._relevant = tmp
            tmp = np.ndarray(mr - 1)
            tmp[:r] = self.alpha[:r]
            tmp[r:] = self.alpha[r + 1:]
            self._alpha = tmp
            self.used[max_i] = -1
            self._fix_used()
        else:
            print 'Uknown Action'
            return 0.
        #print 'RELEVANT: ', self.relevant
        #print 'ALPHA: ', self.alpha
        self._fix_relevant_gsvd()
        return max_delta_log_like

    def _get_log_like(self, beta):
        """Get the log likelihood at beta."""
        prev_beta = self.beta
        self._beta = beta
        self._fix_relevant_gsvd()
        self._compute_statistics()
        self._beta = prev_beta
        return self.log_like

    def _update_beta(self):
        """Update the current value of beta."""
        self._compute_statistics()
        prev_beta = self.beta
        prev_log_like = self._log_like
        #print self.weights.shape
        #print self.PHIs[:, self.relevant].shape
        tmpm = self.Y - np.dot(self.PHIs[:, self.relevant], self.weights)

        sum_a_Lambda = np.einsum('ij, ij, i', self.sigma_sqrt,
                                 self.sigma_sqrt, self.alpha)
        self._beta = ((self.num_samples - self.num_relevant + sum_a_Lambda) /
                      (np.einsum('ij, ij', tmpm, tmpm) / self.num_output))
        self._fix_relevant_gsvd()
        self._compute_statistics()
        if self.log_like < prev_log_like:
            self._beta = prev_beta
            self._fix_relevant_gsvd()
            self._compute_statistics()

    def __init__(self):
        """Initialize the object.

        Does nothing.
        """
        pass

    def set_data(self, PHI, Y):
        assert isinstance(PHI, np.ndarray)
        self._PHI = self._to_2d(PHI)
        assert isinstance(Y, np.ndarray)
        self._Y = self._to_2d(Y)
        assert self.Y.shape[0] == self.PHI.shape[0]
        self._allocate_memory()
        self._scale_PHI()

    def initialize(self, beta=None, relevant=None, alpha=None):
        """Initialize the training algorithm.

        If no, parameters are specified then, they are found
        automatically.
        """
        if beta is None:
            # Use the variance of the data
            beta = self.num_output / np.var(self.Y, axis=0).sum()
        beta = float(beta)
        assert beta > 0.
        if not relevant is None:
            # Initialize at an arbitrary point
            relevant = np.array(relevant, dtype='uint')
            assert relevant.ndim == 1
            assert not alpha is None
            alpha = np.ndarray(alpha)
            assert alpha.ndim == 1
            assert (alpha > 0.).all()
        else:
            # Initialize by looking at the maximum change in marg. like.
            tmpm = np.dot(self.PHIs.T, self.Y)
            tmpv = np.sqrt(np.einsum('ij, ij->j', tmpm, tmpm))
            idx = np.argmax(tmpv)
            pmax = tmpv[idx]
            test_y_max = tmpv[idx]
            alpha_idx = 1. / (test_y_max - 1. / beta)
            relevant = np.array([idx], dtype='uint')
            alpha = np.array([alpha_idx])
        self._relevant = relevant.copy()
        self._alpha = alpha.copy()
        self._beta = beta
        self._fix_used()
        self._fix_relevant_gsvd()
        self._log_like = -1e99

    def train(self, max_it=10000, tol=1e-6, verbose=False):
        """Train the model.

        Keyword Arguments:
            max_it      ---     Maximum number of iterations.
            tol         ---     Desired tolerance.
            verbose     ---     Be verbose or not.

        Note:
            The object must be initialized.
        """
        for i in xrange(max_it):
            prev_log_like = self.log_like
            delta_max_log_like = self._step()
            #print self.relevant
            #print self.alpha
            if i % 1 == 0 and verbose:
                s = str(i) + ': m = ' + str(self.num_relevant)
                s += ', log_like = ' + str(self.log_like)
                s += ', delta_log_like = ' + str(delta_max_log_like)
                s += ', ' + str(prev_log_like + delta_max_log_like)
                print s
            if math.fabs(self.log_like - prev_log_like) < tol:
                prev_log_like = self.log_like
                prev_beta = self.beta
                self._update_beta()

                if verbose:
                    print '** New beta: ', self.beta
                if math.fabs(self.log_like - prev_log_like) < tol:
                    if verbose:
                        print '*** Converged!'
                    break
        self._finalize()

    def get_generalized_linear_model(self, basis):
        """Construct a generalized linear model.

        Arguments:
            basis       ---     The basis you used to construct the
                                design matrix.

        Return:
            A generalized linear model.
        """
        if isinstance(basis, best.maps.CovarianceFunctionBasis):
            sp_basis = best.maps.CovarianceFunctionBasis(basis.cov,
                                                          basis.X[self.relevant, :])
        else:
            sp_basis = basis.screen(out_idx=self.relevant)
        return best.maps.GeneralizedLinearModel(sp_basis, weights=self.weights,
                                                sigma_sqrt=self.sigma_sqrt,
                                                beta=self.beta)