"""Sparse Pseudo-Input Gaussian Process.

Author:
    Ilias Bilionis

Date:
    2/21/2013

"""


__all__ = ['SparsePseudoInputGaussianProcess']


import numpy as np
import math
from scipy.optimize import fmin_bfgs
from scipy.cluster.vq import *
from ._spgp import *
from ..maps import Function


class SparsePseudoInputGaussianProcess(Function):
    """Sparse Pseudo-Input Gaussian Process."""

    # Total number of observations
    _n = None

    # Number of training points
    _n_train = None

    # Number of points for prediciton
    _n_predict = None

    # Number of pseudo inputs
    _m = None

    # Input points
    _X = None

    # Output points
    _Y = None

    # Maximum number of iterations while training
    _maxiter = None

    # Verbosity
    _verbose = None

    @property
    def n(self):
        return self._n

    @property
    def n_train(self):
        return self._n_train

    @property
    def n_predict(self):
        return self._n_predict

    @property
    def m(self):
        return self._m

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y

    @property
    def maxiter(self):
        return self._maxiter

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        assert isinstance(value, bool)
        self._verbose = value

    def __init__(self, X, Y, n_train, m, maxiter=10000, verbose=False):
        """Initialize the object.

        Arguments:
        X               ---         The input.
        Y               ---         The output.
        """
        assert isinstance(X, np.ndarray)
        self._n = X.shape[0]
        self._n_predict = self.n
        self._X = X
        self._n_train = n_train
        self._m = m
        assert isinstance(Y, np.ndarray)
        self._Y = Y
        super(SparsePseudoInputGaussianProcess, self).__init__(X.shape[1],
                                                               Y.shape[1])
        self._maxiter = maxiter
        self.verbose = verbose

    def _train_single_output(self, i):
        """Train a single output."""
        hyp_init = np.hstack(
            [-2. * np.log((np.max(self.X, 0) - np.min(self.X, 0)).T / 2.),
             math.log(np.var(self.Y[:, i])),
             math.log(np.var(self.Y[:, i]) / 10.)])
        w_init = np.hstack([self._xb_init.flatten(order='F'), hyp_init])
        # Likelihood function wrapper
        def f(w, args):
            return spgp_lik(w, args['y'], args['x'], args['M'], d=args['d'],
                          compute_der=False)
        # The derivative of the likelihood
        def fp(w, args):
            return spgp_lik(w, args['y'], args['x'], args['M'], d=args['d'],
                            compute_der=True)[1]
        # Zero the mean of the data
        me_y = np.mean(self.Y[:, i])
        Y0 = self.Y[:self.n_train, i:(i + 1)] - me_y
        data = {}
        data['x'] = self.X[:self.n_train, :]
        data['y'] = Y0
        data['M'] = self.m
        data['d'] = 1e-6
        w, fopt, gopt, Bopt, func_calls, grad_calls, warnflag = fmin_bfgs(f, w_init,
                                                                          fprime=fp,
                                                                          args=(data, ),
                                                                          full_output=True,
                                                                          disp=True,
                                                                          maxiter=self.maxiter)
        return me_y, w

    def train(self):
        """Train model."""
        self._xb_init, dist = kmeans(self.X, self.m)
        me_y_all = []
        w_all = []
        # Train each output individually.
        for i in range(self.num_output):
            if self.verbose:
                print 'Training output, ', i
            me_y, w = self._train_single_output(i)
            me_y_all.append(me_y)
            w_all.append(np.atleast_2d(w))
        self.me_y_all = np.hstack(me_y_all)
        self.w_all = np.vstack(w_all)

    def _evaluate_single_output(self, i, x):
        """Evaluate the i-th output at x."""
        xb = self.w_all[i, :(self.m * self.num_input)].reshape(
            (self.m, self.num_input), order='F')
        hyp = self.w_all[i, (self.m * self.num_input):]
        mu, s2 = spgp_pred(self.Y[:self.n_predict, i:(i + 1)],
                self.X[:self.n_predict, :], xb, x, hyp)
        mu += self.me_y_all[i]
        return mu, s2

    def __call__(self, x, return_s2=False):
        """Evaluate the model at x."""
        # Evaluate each output:
        MU = []
        S2 = []
        for i in range(self.num_output):
            mu, s2 = self._evaluate_single_output(i, x)
            MU.append(mu)
            S2.append(s2)
        MU = np.hstack(MU)
        S2 = np.hstack(S2)
        if return_s2:
            return MU, S2
        else:
            return MU
