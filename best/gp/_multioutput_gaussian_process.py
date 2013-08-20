"""Define the Multioutput Gaussian Process model.

Author:
    Ilias Bilionis

Date:
    11/20/2012
    1/29/2013   (Made it conform to MarkovChainMonteCarlo class so that it can
                 be used in conjuction with uq.random.SequentialMonteCarlo.)
    1/31/2013   (Added sample_prediction.)
    1/2/2013    (Added add_data and sample_surrogate.)
"""


__all__ = ['MultioutputGaussianProcess']


import math
import numpy as np
import scipy.linalg
from scipy.stats import lognorm
import scipy.sparse as sp
import itertools as iter
from ..linalg import kron_prod
from ..linalg import incomplete_cholesky
from ..linalg import kron_solve
from ..linalg import update_cholesky
from ..linalg import update_cholesky_linear_system
#from ..linalg import update_qr
from . import SECovarianceFunction
from . import SeparableCovarianceFunction
from ..random import Distribution
from ..random import LikelihoodFunction
from ..random import PosteriorDistribution
from ..random import ProposalDistribution
from ..random import MarkovChainMonteCarlo
#import matplotlib.pyplot as plt


# Default values for the parameters
MGP_DEFAULT_NUM_MCMC = 1
MGP_DEFAULT_NUM_INIT = 1
MGP_DEFAULT_SAMPLE_G = True
MGP_DEFAULT_SAMPLE_R = True
MGP_DEFAULT_R = 0.1
MGP_DEFAULT_G = 1e-2
MGP_DEFAULT_SIGMA_R = 1e-2
MGP_DEFAULT_SIGMA_G = 1e-2
MGP_DEFAULT_GAMMA = 0.1
MGP_DEFAULT_DELTA = 1e-4


class _Prior(Distribution):
    """A simple prior distribution for the hyper-parameters."""

    # Parameter of the exponential distribution
    _delta = None

    # The number of dimensions per component
    _k_of = None

    # The number of length scales
    _k = None

    # The number of components
    _s = None

    @property
    def delta(self):
        """Get the parameers of the expoential distribution."""
        return self._delta

    @property
    def k_of(self):
        """Get the number of dimensions per component."""
        return self._k_of

    @property
    def k(self):
        """Get the number of length scales."""
        return self._k

    @property
    def s(self):
        """Get the number of components."""
        return self._s

    def __init__(self, k_of, name='GP Prior for hyper-parameters'):
        """Initialize the object.

        Arguments:
            k_of        ---     The dimension of each of the s spaces.
        """
        self._s = len(k_of)
        self._k_of = k_of
        self._k = np.sum(k_of)
        num_input = self.k + self.s
        super(_Prior, self).__init__(num_input, name)
        self._delta =  np.array([MGP_DEFAULT_GAMMA] * self.s
                + [MGP_DEFAULT_DELTA] * self.s)

    def sample(self):
        """Sample the prior."""
        r = []
        for i in range(self.s):
            r += [np.random.exponential(scale=self.delta[i],
                size=self.k_of[i])]
        g = []
        for i in range(self.s):
            g += [np.random.exponential(scale=self.delta[self.s + i])]
        x = np.hstack(r + g)
        return x


class _Likelihood(LikelihoodFunction):
    """A fake likelihood function for the GP's."""

    def __init__(self, k_of, name='A Fake likelihood Function for GP'):
        """Initalize the object."""
        num_input = np.sum(k_of) + len(k_of)
        super(_Likelihood, self).__init__(num_input, name)


class _Posterior(PosteriorDistribution):
    """A fake posterior distribution for the GP's."""

    # A pointer to the MultioutputGaussianProcess class
    _mgp = None

    def __init__(self, mgp, likelihood=None, prior=None):
        """Initialize the object.

        Arguments:
            mgp     ----    Simply provides access to the core
                            MultioutputGaussianProcess class.
        """
        if not isinstance(likelihood, _Likelihood):
            raise TypeError('The likelihood should only be a _Likelihood.')
        if not isinstance(prior, _Prior):
            raise TypeError('The prior should only be a _Prior.')
        super(_Posterior, self).__init__(likelihood=likelihood, prior=prior)
        if not isinstance(mgp, MultioutputGaussianProcess):
            raise TypeError('mgp must be a MultioutputGaussianProcess.')
        self._mgp = mgp

    def __call__(self, x, report_all=False):
        """Evaluates the posterior at x.

        Arguments:
            x       ---     The point of evaluation

        Keyword Arguments:
            report_all      ---     If set to False, then the function simply
                                    returns the log of the posterior.
                                    Otherwise, it returns a dictionary
                                    representating the state of the object.
        """
        self._mgp.initialize(x, eval_state=None)
        if report_all:
            return self._mgp.current_state
        else:
            return self._mgp.log_post_lk


class _Proposal(ProposalDistribution):
    """A simple class that represents the proposal."""

    @property
    def dt(self):
        """Get the step of the proposal."""
        return self._dt

    @dt.setter
    def dt(self, value):
        """Set the proposal step."""
        assert isinstance(value, np.ndarray)
        self._dt = value


class MultioutputGaussianProcess(MarkovChainMonteCarlo):

    """Define the MultioutputGaussianProcess class."""

    # The name of the model
    _name = None

    # The covariance function (imutable)
    _cov = None

    # Number of MCMC parameters per Gibbs step
    _num_mcmc = None

    # Number of initializations
    _num_init = None

    # Number of components
    _s = None

    # Number of samples (imutable)
    _n = None

    # Number of basis functions (imutable)
    _m = None

    # Number of inputs
    _k = None

    # Number of outputs
    _q = None

    # Number of samples per component (imutable, tuple)
    _n_of = None

    # Number of basis functions per compponent (imutable, tuple)
    _m_of = None

    # Number of dimensions per component (imutable, tuple)
    _k_of = None

    # Output data
    _Y = None

    # Input data
    _X = None

    # Design matrices
    _H = None

    # The nuggets
    _g = None
    _gn = None

    # The length scales
    _r = None
    _rn = None

    # The correlation matrix
    _Sigma = None
    _Sigman = None

    # The Cholesky decomposition of Sigma
    _LSigma = None
    _LSigman = None

    # The weights of the mean
    _B = None
    _Bn = None

    # The cholesky decomposition of the covariance matrices
    _L_A = None
    _L_An = None

    # The determinants of the covariance matrices
    _log_det_A = None
    _log_det_An = None

    # The scaled design matrices
    _Hs = None
    _Hsn = None

    # The QR factorization of the scaled design matrices
    _QLAiH = None
    _QLAiHn = None
    _RLAiH = None
    _RLAiHn = None

    # The log of determinans of the R parts of H^TA^(-1)H
    _log_det_HTAiH = None
    _log_det_HTAiHn = None

    # The log of the determinant of Sigma
    _log_det_Sigma = None
    _log_det_Sigman = None

    # The logarithm of the posterior likelihood
    _log_post_lk = None

    # A scaled version of the data
    _Ys = None
    _Ysn = None

    # A centered version of the scaled version of the data
    _YmHBs = None
    _YmHBsn = None

    # Acceptance tate per component
    _acc_rate = None
    _acc_count = None
    _g_acc_rate = None
    _g_acc_count = None

    # Do you want to sample g?
    _sample_g = None

    # Do you want to sample r?
    _sample_r = None

    # For Sequential Monte Carlo
    # The log likelihood of the current state
    _log_like = None

    @property
    def log_like(self):
        """Get the log likelihood of the current state."""
        return self._log_like

    @property
    def name(self):
        """Get the name of the model."""
        return self._name

    @name.setter
    def name(self, value):
        """Set the name of the model."""
        if not isinstance(value, str):
            raise TypeError('The name must be a string.')
        self._name = value

    @property
    def sample_g(self):
        """Ask if g has to be sampled."""
        return self._sample_g

    @sample_g.setter
    def sample_g(self, value):
        """Set sample_g."""
        if not isinstance(value, bool):
            raise TypeError('A boolean is needed.')
        self._sample_g = value

    @property
    def sample_r(self):
        """Ask if r has to be sampled."""
        return self._sample_r

    @sample_r.setter
    def sample_r(self, value):
        """Set sample_r."""
        if not isinstance(value, bool):
            raise TypeError('A boolean is needed.')
        self._sample_r = value

    @property
    def cov(self):
        """Get the covariance function."""
        return self._cov

    @property
    def num_mcmc(self):
        """Get the number of MCMC parameters per Gibbs step."""
        return self._num_mcmc

    @num_mcmc.setter
    def num_mcmc(self, value):
        """Set the number of MCMC parameters per Gibbs step."""
        if not isinstance(value, int):
            raise TypeError('num_mcmc must be an int.')
        if value <= 0:
            raise TypeError('num_mcmc must be postive.')
        self._num_mcmc = value

    @property
    def gamma(self):
        """Get the prior parameters for."""
        return self.target.prior.delta[:self.s]

    @property
    def delta(self):
        """Get the prior paramters for g."""
        return self.target.prior.delta[self.s:]

    @property
    def sigma_r(self):
        """Get the proposal step for r."""
        return self.proposal.dt[:self.s]

    @property
    def sigma_g(self):
        """Get the proposal step for g."""
        return self.proposal.dt[self.s:]

    @property
    def num_init(self):
        """Get the number of initialization."""
        return self._num_init

    @num_init.setter
    def num_init(self, value):
        """Set the number of initializations."""
        if not isinstance(value, int):
            raise TypeError('num_init must be an int.')
        if value <= 0:
            raise TypeError('num_init must be positive.')
        self._num_init = value

    @property
    def n(self):
        """Get the number of samples."""
        return self._n

    @property
    def m(self):
        """Get the number of basis functions."""
        return self._m

    @property
    def k(self):
        """Get the number of inputs."""
        return self._k

    @property
    def q(self):
        """Get the number of outputs."""
        return self._q

    @property
    def n_of(self):
        """Get the number of samples per component."""
        return self._n_of

    @property
    def k_of(self):
        """Get the number of dimensions per component."""
        return self._k_of

    @property
    def m_of(self):
        """Get the number of basis functions per component."""
        return self._m_of

    @property
    def s(self):
        """Get the number of separable components."""
        return self._s

    @property
    def X(self):
        """Get the input data."""
        return self._X

    @property
    def Y(self):
        """Get the output data."""
        return self._Y

    @property
    def H(self):
        """Get the design matrices."""
        return self._H

    @property
    def g(self):
        """Get the nuggets."""
        return self._g

    @property
    def r(self):
        """Get the length scales."""
        return self._r

    @property
    def Sigma(self):
        """Get the correlation matrix."""
        return self._Sigma

    @property
    def log_post_lk(self):
        """Get the logarithm of the posterior likelihood."""
        return self._log_post_lk

    @property
    def acc_rate(self):
        """Get the acceptance rate."""
        return self._acc_rate

    @property
    def g_acc_rate(self):
        """Get the acceptance rate of the nuggets."""
        return self._g_acc_rate

    @property
    def acceptance_rate(self):
        """Get the acceptance rate of everything."""
        try:
            return np.hstack([self._acc_rate / self._acc_count,
                self._g_acc_rate / self._g_acc_count])
        except ArithmeticError, err:
            return np.zeros(2 * self.s)

    def __str__(self):
        """Return a string representation of the object."""
        s = self.name
        s += '\n' + str(self.cov)
        s += '\nParameters:'
        s += '\nnum_mcmc: ' + str(self.num_mcmc)
        s += '\ngamma: ' + str(self.gamma)
        s += '\ndelta: ' + str(self.delta)
        s += '\nsigma_r: ' + str(self.sigma_r)
        s += '\nsigma_g: ' + str(self.sigma_g)
        s += '\nnum_init: ' + str(self.num_init)
        if self.X is not None:
            s += '\nComponents:'
            s += '\nk: ' + str(self.k_of)
            s += '\nn: ' + str(self.n_of)
            s += '\nm: ' + str(self.m_of)
            s == '\nq: ' + str(self.q)
        return s

    def __init__(self, mgp=None, name='MultioutputGaussianProcess'):
        """Initialize the object.

        This doesn't do much.
        """
        self.name = name
        if mgp is not None:
            self._cov = mgp.cov
            self._num_init = mgp.num_init
            self._num_mcmc = mgp.num_mcmc
            self._sample_r = mgp.sample_r
            self._sample_g = mgp.sample_g
            self._proposal = mgp.proposal
            self._target = mgp.target
        else:
            self.num_mcmc = MGP_DEFAULT_NUM_MCMC
            self.num_init = MGP_DEFAULT_NUM_INIT
            self.sample_g = MGP_DEFAULT_SAMPLE_G
            self.sample_r = MGP_DEFAULT_SAMPLE_R

    def set_data(self, X, H, Y):
        """Set the observation data."""
        if (isinstance(X, np.ndarray) and isinstance(H, np.ndarray) and
            isinstance(Y, np.ndarray) and not isinstance(X, tuple)):
            X = (X, )
            H = (H, )
        if not len(X) == len(H):
            raise ValueError('The dimensions of X and H must much.')
        for x, h in iter.izip(X, H):
            if not isinstance(x, np.ndarray):
                raise TypeError('Each component of X must be a numpy array.')
            if not len(x.shape) == 2:
                raise ValueError(
                    'Each component of X must be two dimensional.')
            if not isinstance(h, np.ndarray):
                raise TypeError('Each component of H must be a numpy array.')
            if not len(h.shape) == 2:
                raise ValueError(
                        'Each component of H must be two dimensional.')
        if not isinstance(Y, np.ndarray):
            raise TypeError('Y must be a numpy array.')
        if len(Y.shape) == 1:
            Y = Y.reshape((Y.shape[0], 1))
        if not len(Y.shape) == 2:
            raise TypeError('Y must be either 1 or 2 dimensional array.')
        self._X = X
        self._H = H
        self._Y = Y
        self._get_dimensions()
        self._allocate_memory()
        self._set_default_parameters()

    def _set_default_parameters(self):
        """Set the hyper-parameters to some default values."""
        self.sigma_r.fill(MGP_DEFAULT_SIGMA_R)
        self.sigma_g.fill(MGP_DEFAULT_SIGMA_G)
        self.gamma.fill(MGP_DEFAULT_GAMMA)
        self.delta.fill(MGP_DEFAULT_DELTA)

    def _get_dimensions(self):
        """Set the dimensions and check for correctness.

        Assumes X, H and Y have been set.
        """
        self._s = len(self._X)
        self._q = self.Y.shape[1]
        self._k_of = tuple(x.shape[1] for x in self.X)
        self._n_of = tuple(x.shape[0] for x in self.X)
        assert self._n_of == tuple(h.shape[0] for h in self.H)
        self._m_of = tuple(h.shape[1] for h in self.H)
        self._k = sum(self.k_of)
        self._n = np.prod(self.n_of)
        assert self.Y.shape[0] == self.n
        self._m = np.prod(self.m_of)

    def _allocate_memory(self):
        """Allocate memory.

        Assumes that _get_dimensions has been called.
        """
        # Allocate the covariance functions
        c = tuple(SECovarianceFunction(k) for k in self.k_of)
        self._cov = SeparableCovarianceFunction(cov=c)
        # The rest of the data
        self._acc_rate = np.ndarray(self.s)
        self._acc_count = np.ndarray(self.s)
        self._g_acc_rate = np.ndarray(self.s)
        self._g_acc_count = np.ndarray(self.s)
        self._r = [np.ndarray(k, order='F') for k in self.k_of]
        self._rn = [np.ndarray(k, order='F') for k in self.k_of]
        self._LA = [np.ndarray((n, n), order='F') for n in self.n_of]
        self._LAn = [np.ndarray((n, n), order='F') for n in self.n_of]
        self._Hs = [np.ndarray((n, m), order='F')
                for n, m in iter.izip(self.n_of, self.m_of)]
        self._Hsn = [np.ndarray((n, m), order='F')
                for n, m in iter.izip(self.n_of, self.m_of)]
        self._QLAiH = [np.ndarray((n, n), order='F') for n in self.n_of]
        self._QLAiHn = [np.ndarray((n, n), order='F') for n in self.n_of]
        self._RLAiH = [np.ndarray((n, m), order='F')
                for n, m in iter.izip(self.n_of, self.m_of)]
        self._RLAiHn = [np.ndarray((n, m), order='F')
                for n, m in iter.izip(self.n_of, self.m_of)]
        self._Sigma = np.ndarray((self.q, self.q), order='F')
        self._Sigman = np.ndarray((self.q, self.q), order='F')
        self._LSigma = np.ndarray((self.q, self.q), order='F')
        self._LSigman = np.ndarray((self.q, self.q), order='F')
        self._g = np.ndarray(self.s, order='F')
        self._gn = np.ndarray(self.s, order='F')
        self._log_det_A = np.ndarray(self.s, order='F')
        self._log_det_An = np.ndarray(self.s, order='F')
        self._log_det_HTAiH = np.ndarray(self.s, order='F')
        self._log_det_HTAiHn = np.ndarray(self.s, order='F')
        self._B = np.ndarray((self.m, self.q), order='F')
        self._Bn = np.ndarray((self.m, self.q), order='F')
        self._Ys = np.ndarray((self.n, self.q), order='F')
        self._Ysn = np.ndarray((self.n, self.q), order='F')
        self._YmHBs = np.ndarray((self.n, self.q), order='F')
        self._YmHBsn = np.ndarray((self.n, self.q), order='F')
        prior = _Prior(self.k_of)
        likelihood = _Likelihood(self.k_of)
        self._target = _Posterior(self, likelihood=likelihood, prior=prior)
        self._proposal = _Proposal(dt=np.ndarray(2 * self.s))

    def _compute_inner_dep_of(self, i, LA, Hs, QLAiH, RLAiH):
        """Compute inner i dependencies."""
        # Compute the determinant of A
        log_det_A = 2. * np.log(np.diag(LA)).sum()
        # Solve the linear system LA * Hs = H
        #Hs[:] = self._H[i][:]
        #scipy.linalg.solve_triangular(LA, Hs, lower=True, overwrite_b=True)
        #trsm(LA, Hs)
        Hs[:] = scipy.linalg.solve_triangular(LA, self._H[i], lower=True)
        # Compute the QR factorization of Hs
        QLAiH[:], RLAiH[:] = scipy.linalg.qr(Hs, mode='full')
        # Compute the determinant of HTAiH
        log_det_HTAiH = 2. * np.log(np.abs(np.diag(RLAiH))).sum()
        return log_det_A, log_det_HTAiH

    def _compute_dep_of(self, i, r, g, LA, Hs, QLAiH, RLAiH):
        """Compute i dependent."""
        # Calculate the covariance matrix for LA
        self.cov.cov[i](r, self.X[i], A=LA)
        # Add the nuggets
        LA += g * np.eye(LA.shape[0])
        d = np.linalg.det(LA)
        # Compute the Cholesky decomposition (in place)
        scipy.linalg.cholesky(LA, lower=True, overwrite_a=True)
        log_det_A, log_det_HTAiH = self._compute_inner_dep_of(i, LA, Hs, QLAiH,
                                                              RLAiH)
        return log_det_A, log_det_HTAiH

    def _compute_dep(self, LA, Hs, QLAiH, RLAiH, Ys, B, YmHBs, Sigma, LSigma):
        """Compute partial dependences."""
        # Compute Ys
        #Ys[:] = self.Y[:]
        #kron_trsm(LA, Ys)
        Ys[:] = kron_solve(LA, self.Y[:])
        #Y = kron_prod(LA, Ys)
        #print Y - self.Y

        # Compute B
        QT = ()
        for Q in QLAiH:
            QT += (Q.T, )
        tmpm_n_q = kron_prod(QT, Ys)
        B[:] = tmpm_n_q[:self.m, :self.q]
        R = ()
        for m, Rc in iter.izip(self.m_of, RLAiH):
            R += (Rc[:m, :m], )
        uplo = 'U' * self.s
        #kron_trsm(R, B, uplo=uplo)
        B[:] = kron_solve(R, B)
        # Compute Sigma
        YmHBs[:] = Ys - kron_prod(Hs, B)
        Sigma[:] = (np.dot(YmHBs.T, YmHBs)) / (self.n - self.m)

        # Compute the Cholesky decomposition of Sigma
        LSigma[:] = scipy.linalg.cholesky(Sigma, lower=True)

        # Compute the determinant of Sigma
        log_det_Sigma = 2. * np.log(np.diag(LSigma)).sum()

        return log_det_Sigma

    def _compute_log_post_lk(self):
        """Compute the posterior likelihood."""
        p0 = 0.
        p1 = 0.
        p2 = 0.
        for i in xrange(self.s):
            p0 -= self._g[i] / self.delta[i]
            p0 -= (self._r[i] / self.gamma[i]).sum()
            p1 -= 0.5 * (self.n / self.n_of[i]) * self.q * self._log_det_A[i]
            p2 -= (0.5 * (self.m / self.m_of[i]) * self.q *
                   self._log_det_HTAiH[i])
        p3 = - 0.5 * (self.n - self.m) * self._log_det_Sigma
        # The log-likelihood is p1 + p2 + p3
        log_like = p1 + p2 + p3
        return p0 + self.target.gamma * (p1 + p2 + p3), log_like

    def _compute_all_dep(self):
        """Compute all dependences."""
        for i in xrange(self.s):
            self._log_det_A[i], self._log_det_HTAiH[i] = self._compute_dep_of(
                        i, self._r[i], self._g[i], self._LA[i], self._Hs[i],
                        self._QLAiH[i], self._RLAiH[i])
        self._log_det_Sigma = self._compute_dep( self._LA, self._Hs,
                        self._QLAiH, self._RLAiH,
                        self._Ys, self._B, self._YmHBs, self._Sigma,
                        self._LSigma)
        self._log_post_lk, self._log_like = self._compute_log_post_lk()

    def initialize(self, hyp, eval_state=None):
        """Initialize the object.

        Argument:
        hyp     ---     The initial hyper-parameters. It must be a numpy array
                        ordered so that the first self.k elements correspond to
                        the length scales and the last self.s to the nuggets.
                        The length scales are ordered so that the first
                        self.k_of[0] correspond to the first group of input
                        variables, the following self.k_of[1] correspond to the
                        second group and so on.

        Keyword Arguments:
            eval_state      ---     A dictionary that contains all the data
                                    required to start the MCMC algorithm from
                                    the specified hyper-parameters. If not
                                    specified, then these data are initialized
                                    from scratch. The correct format of eval
                                    state is the one returned by self.sample().
        """
        assert isinstance(hyp, np.ndarray)
        assert len(hyp.shape) == 1 and hyp.shape[0] == self.k + self.s
        k_so_far = 0
        for i in xrange(self.s):
            self._r[i][:] = hyp[ k_so_far:(k_so_far + self.k_of[i])]
            k_so_far += self.k_of[i]
            self._g[i] = hyp[-self.s + i]
        if eval_state is not None:
            self._log_like = eval_state['log_like']
            self._log_post_lk = eval_state['log_p']
            self._log_det_A[:] = eval_state['log_det_A']
            self._log_det_HTAiH[:] = eval_state['log_det_HTAiH']
            for t, f in iter.izip(self._LA, eval_state['LA']):
                t[:] = f
            for t, f in iter.izip(self._Hs, eval_state['Hs']):
                t[:] = f
            for t, f in iter.izip(self._QLAiH, eval_state['QLAiH']):
                t[:] = f
            for t, f in iter.izip(self._RLAiH, eval_state['RLAiH']):
                t[:] = f
            self._B[:] = eval_state['B']
            self._Ys[:] = eval_state['Ys']
            self._YmHBs[:] = eval_state['YmHBs']
            self._Sigma[:] = eval_state['Sigma']
            self._LSigma[:] = eval_state['L_Sigma']
        else:
            self._compute_all_dep()
        self._acc_rate[i] = 0.
        self._acc_count[i] = 0.
        self._g_acc_rate[i] = 0.
        self._g_acc_count[i] = 0.

    def _propose_r(self, i, r, rn):
        """Propose a new r."""
        rn[:] = np.exp(np.log(r) + self.sigma_r[i] * np.random.randn(*r.shape))

    def _propose_g(self, i, g):
        """Propose a new nugget."""
        try:
            u = np.random.randn()
            new_g = math.exp(math.log(g) + self.sigma_g[i] * u)
        except:
            print 'i: ', i
            print 'u: ', u
            print 'g:', g
            print 'sigma:', self.sigma_g[i]
            print 'log(g):', math.log(g)
            self.comm.abort(0)
        return math.exp(math.log(g) + self.sigma_g[i] * np.random.randn())

    def _compute_log_a1(self, i, r, g):
        """Compute the part of the acceptance ratio."""
        # Compute the statistics that are relevant only to i.
        self._log_det_An[i], self._log_det_HTAiHn[i] = (
                self._compute_dep_of(i, r, g,
                self._LAn[i], self._Hsn[i], self._QLAiHn[i],
                self._RLAiHn[i]))
        self._LAn[i], self._LA[i] = self._LA[i], self._LAn[i]
        self._Hsn[i], self._Hs[i] = self._Hs[i], self._Hsn[i]
        self._QLAiHn[i], self._QLAiH[i] = (
                self._QLAiH[i], self._QLAiHn[i])
        self._RLAiHn[i], self._RLAiH[i] = (
                self._RLAiH[i], self._RLAiHn[i])

        # Compute the statistics that depend on i and everybody else
        self._log_det_Sigman = self._compute_dep(self._LA, self._Hs,
                                                 self._QLAiH, self._RLAiH,
                                                 self._Ysn, self._Bn,
                                                 self._YmHBsn,
                                                 self._Sigman,
                                                 self._LSigman)
        # Part 1 from the posterior
        p0 = (-((r - self._r[i]) / self.gamma[i]).sum()
              -((g - self._g[i]) / self.delta[i]))

        # Contribution from the determinant of A[i]
        p1 = -0.5 * (self.n / self.n_of[i]) * self.q * (
                self._log_det_An[i] - self._log_det_A[i])

        # Contribution from the determinant of H[i]^T*A[i]^{-1}*H[i]:
        p2 = -0.5 * (self.m / self.m_of[i]) * self.q * (
                self._log_det_HTAiHn[i] - self._log_det_HTAiH[i])

        # Contribution from the determinant of Sigma:
        p3 = -0.5 * (self.n - self. m) * (
                self._log_det_Sigman - self._log_det_Sigma)

        log_a1 = p0 + self.target.gamma * (p1 + p2 + p3)

        return log_a1

    def _swap_accept(self, i):
        """Swap the relevant data when step is accepted."""
        self._log_det_A[i] = self._log_det_An[i]
        self._log_det_HTAiH[i] = self._log_det_HTAiHn[i]
        self._log_det_Sigma = self._log_det_Sigman
        self._Ysn, self._Ys = self._Ys, self._Ysn
        self._YmHBsn, self._YmHBs = self._YmHBs, self._YmHBsn
        self._Bn, self._B = self._B, self._Bn
        self._Sigman, self._Sigma = self._Sigma, self._Sigman
        self._LSigman, self._LSigma = self._LSigma, self._LSigman

    def _swap_reject(self, i):
        """Swap all relevant data when move is rejected."""
        self._LAn[i], self._LA[i] = self._LA[i], self._LAn[i]
        self._Hsn[i], self._Hs[i] = self._Hs[i], self._Hsn[i]
        self._QLAiHn[i], self._QLAiH[i] = (
                self._QLAiH[i], self._QLAiHn[i])
        self._RLAiHn[i], self._RLAiH[i] = (
                self._RLAiH[i], self._RLAiHn[i])

    def _sample_gibbs(self, term='r'):
        """Perform one MCMC step to sample the length scales."""
        # Loop over components - possibly repeatedly
        comp = [i for i in range(self.s) for _ in range(self.num_mcmc)]
        for i in comp:
            try:
                if term == 'r':
                    self._acc_count[i] += 1
                    self._propose_r(i, self._r[i], self._rn[i])
                    try:
                        log_a1 = self._compute_log_a1(i, self._rn[i], self._g[i])
                    except:
                        log_a1 = -1e99
                    try:
                        log_a2 = (np.log(self._rn[i]) - np.log(self._r[i])).sum()
                    except:
                        log_a2 = -1e99
                else:
                    self._g_acc_count[i] += 1
                    self._gn[i] = self._propose_g(i, self._g[i])
                    try:
                        log_a1 = self._compute_log_a1(i, self._r[i], self._gn[i])
                    except:
                        log_a1 = -1e99
                    try:
                        log_a2 = math.log(self._gn[i]) - math.log(self._g[i])
                    except:
                        log_a2 = -1e99
                log_a = log_a1 + log_a2
                a = 1. if log_a >= 0. else math.exp(log_a)
            except scipy.linalg.LinAlgError, err:
                # Rejecting move because Cholesky failed!
                a = 0.

            # Accept or reject
            u = np.random.rand()
            if u <= a:
                if term == 'r':
                    self._acc_rate[i] += 1
                    self._rn[i], self._r[i] = self._r[i], self._rn[i]
                else:
                    self._g_acc_rate[i] += 1
                    self._gn[i], self._g[i] = self._g[i], self._gn[i]
                self._swap_accept(i)
            else:
                self._swap_reject(i)
            self._log_post_lk, self._log_like = self._compute_log_post_lk()

    def sample(self, x=None, eval_state=None, return_eval_state=False, steps=1):
        """Take samples from the posteriror.

        Keyword Arguments:
            x           ---         The initial state. If not specified, then,
                                    we attemp to use the previous state
                                    processed by this class.
            eval_state  ---         A dictionary containing the all the data
                                    required to initialize the object. Such a
                                    state is actually returned by this
                                    function if the option "return_eval_sate"
                                    is set to True. If not specified, then
                                    everything is calculated from scratch.
            return_eval_state   ---     If specified, then the routine returns
                                        the "evaluated_state" of the sampler,
                                        which may be used to restart the MCMC
                                        sampling.
        """
        if x is not None:
            self.initialize(x)
        for i in xrange(steps):
            if self.sample_r:
                self._sample_gibbs(term='r')
            if self.sample_g:
                self._sample_gibbs(term='g')
        r = np.hstack(self.r)
        g = np.array(self.g)
        current_state = np.hstack([r, g])
        if return_eval_state:
                return current_state, self.current_state
        else:
            return current_state

    @property
    def current_state(self):
        """Return a dictionary representing the current state."""
        eval_state = {}
        eval_state['log_like'] = self.log_like
        eval_state['log_p'] = self.log_post_lk
        eval_state['log_det_A'] = self._log_det_A.copy()
        eval_state['log_det_HTAiH'] = self._log_det_HTAiH.copy()
        eval_state['LA'] = [c.copy() for c in self._LA]
        eval_state['Hs'] = [c.copy() for c in self._Hs]
        eval_state['QLAiH'] = [c.copy() for c in self._QLAiH]
        eval_state['RLAiH'] = [c.copy() for c in self._RLAiH]
        eval_state['B'] = self._B.copy()
        eval_state['Ys'] = self._Ys.copy()
        eval_state['YmHBs'] = self._YmHBs.copy()
        eval_state['Sigma'] = self._Sigma.copy()
        eval_state['L_Sigma'] = self._LSigma.copy()
        return eval_state

    def __getstate__(self):
        """For pickling."""
        state = {}
        state['name'] = self.name
        state['num_mcmc'] = self.num_mcmc
        state['num_init'] = self.num_init
        state['X'] = self.X
        state['Y'] = self.Y
        state['H'] = self.H
        state['r'] = self.r
        state['g'] = self.g
        state['posterior'] = self.target
        state['proposal'] = self.proposal
        return state

    def __setstate__(self, state):
        """For pickling."""
        self.__init__(state['name'])
        self.num_mcmc = state['num_mcmc']
        self.num_init = state['num_init']
        self._target = state['posterior']
        self._proposal = state['proposal']
        self.set_data(state['X'], state['H'], state['Y'])
        r = np.hstack(state['r'])
        g = np.array(state['g'])
        hyp = np.hstack([r, g])
        self.initialize(hyp)

    def __call__(self, X, H, Y=None, C=None, compute_covariance=False):
        """Evaluates the prediction at a given set of points.

        The result of this function, is basically the predictive
        distribution, encoded in terms of the mean Y and the covariance
        matrix C.

        Arguments:
            X   ---     The input points.
            H   ---     The design matrices.

        Keyword Arguments:
            Y   ---     An array to store the mean. If None, then it is
                        returned.
            C   ---     An array to store the covariance. If None, then the
                        covariance is not computed or it is returned as
                        specified by the compute_covariance option.
            compute_covariance  ---     If set to C is None, and the flag
                                        is set to True, then the covariance
                                        is calculated and returned. If C is
                                        not None, then it is ignored.
        """
        if not isinstance(X, tuple) and not isinstance(X, list):
            X = [X]
        if not isinstance(H, tuple) and not isinstance(H, list):
            H = [H]
        if not isinstance(X, list):
            X = list(X)
        if not isinstance(H, list):
            H = list(H)
        for i in xrange(self.s):
            assert X[i].ndim <= 2
            if X[i].ndim == 1:
                X[i] = X[i].reshape((X[i].shape[0], 1))
            assert X[i].shape[1] == self.k_of[i]
            assert H[i].ndim <= 2
            if H[i].ndim == 1:
                H[i] = H[i].reshape((H[i].shape[0], 1))
            assert H[i].shape[1] == self.m_of[i]
        n_of = tuple(x.shape[0] for x in X)
        n = np.prod(n_of)
        return_Y = False
        if Y is None:
            return_Y = True
            Y = np.ndarray((n, self.q), order='F')
        else:
            assert isinstance(Y, np.ndarray)
            assert Y.ndim <= 2
            if Y.ndim == 1:
                Y = Y.reshape((Y.shape[0], 1))
            assert Y.shape[0] == n
            assert Y.shape[1] == self.q
        # Compute the mean
        T = ()
        for i in range(self.s):
            tmp = np.ndarray((self.n_of[i], n_of[i]), order='F')
            self.cov.cov[i](self.r[i], self.X[i], X[i], tmp)
            #trsm(self._LA[i], tmp)
            tmp1 = scipy.linalg.solve_triangular(self._LA[i], tmp,
                                                 lower=True)
            T += (tmp1.T, )
        Y[:] = kron_prod(H, self._B)
        tmp1 = kron_prod(T, self._YmHBs)
        Y += tmp1
        # Compute the covariance
        return_C = False
        if C is None and compute_covariance:
            return_C = True
            C = np.ndarray((n, n), order='F')
        elif C is not None:
            assert isinstance(C, np.ndarray)
            assert C.ndim <= 2
            if C.ndim == 1:
                C = C.reshape((C.shape[0], 1))
            assert C.shape[0] == C.shape[1]
            assert C.shape[0] == n
        if C is not None:
            C0 = ()
            for i in range(self.s):
                C0 += (np.ndarray((n_of[i], n_of[i]), order='F'), )
                self.cov.cov[i](self.r[i], X[i], A=C0[i])
            C00 = C0[0]
            for i in range(1, self.s):
                C00 = np.kron(C00, C0[i])
            C[:] = C00
            C1 = T[0]
            for i in range(1, self.s):
                C1 = np.kron(C1, T[i])
            C -= np.dot(C1, C1.T)
            tmpm = ()
            for i in range(self.s):
                tmpm += (np.dot(self._Hs[i].T, T[i].T), )
            tmpm_3 = tmpm[0]
            tmpm_4 = H[0]
            for i in range(1, self.s):
                tmpm_3 = np.kron(tmpm_3, tmpm[i])
                tmpm_4 = np.kron(tmpm_4, H[i])
            C2 = tmpm_4.T - tmpm_3
            R = ()
            for m, Rc in iter.izip(self.m_of, self._RLAiH):
                R += (Rc[:m, :m], )
            uplo = 'U' * self.s
            C2[:] = kron_solve(R, C2)
            C += np.dot(C2.T, C2)
        if return_Y and return_C:
            return Y, C
        elif return_Y and not return_C:
            return Y
        elif not return_Y and return_C:
            return C

    def sample_prediction(self, X, H, Y=None, C=None):
        """Sample from the predictive distribution of the model.

        Arguments:
            X   ----    The input points.
            H   ----    The design matrices.

        Keyword Arguments:
            Y   ---     An array to store the response. If not specified,
                        then it is allocated and returned.
            C   ---     An optional array that will store the covariance
                        matrix. If not supplied, it will be allocated.
                        On the output, the incomplete Cholesky decomposition
                        is written on C.

        Return:
            If Y is None, then the sample will be returned.
            The trace of the covariance normalized by the number of
            spatial/time inputs and the outputs. This is a measure
            associated with the uncertainty of the given input point.
        """
        n_of = tuple(x.shape[0] for x in X)
        n = np.prod(n_of)
        return_Y = False
        if Y is None:
            return_Y = True
            Y = np.ndarray((n, self.q), order='F')
        if C is None:
            C = np.ndarray((n, n), order='F')
        # Evaluate the posterior
        self(X, H, Y=Y, C=C)
        P, k = incomplete_cholesky(C, in_place=True)
        PC = np.dot(P, C[:, :k])
        unc = np.trace(PC) * np.trace(self.Sigma) / (n *
                np.prod(self.n_of[1:] * self.q))
        z = np.random.standard_t(self.n - self.m, k * self.q)
        #z = np.random.randn(k * self.q)
        Z = kron_prod((PC, self._LSigma), z).reshape((n, self.q))
        Y += Z
        if return_Y:
            return Y, unc
        else:
            return unc

    def add_data(self, X0, H0, Y0):
        """Add more observations to the data set.

        The routine currently only adds observations pertaining to the first
        component. Addition to the other components would ruin the Kronecker
        properties of the matrices.

        Arguments:
            X0      ---     The input variables.
            H0      ---     The design matrix.
            Y0      ---     The observations.
        """
        assert isinstance(X0, np.ndarray)
        assert X0.ndim <= 2
        if X0.ndim == 1:
            X0 = X0.reshape((1, X0.shape[0]))
        assert X0.shape[1] == self.k_of[0]
        assert isinstance(H0, np.ndarray)
        assert H0.ndim <= 2
        if H0.ndim == 1:
            H0 = H0.reshape((1, H0.shape[0]))
        assert X0.shape[0] == H0.shape[0]
        assert H0.shape[1] == self.m_of[0]
        assert isinstance(Y0, np.ndarray)
        assert Y0.ndim <= 2
        if Y0.ndim == 1:
            Y0 = Y0.reshape((1, Y0.shape[0]))
        assert Y0.shape[1] == self.q
        # Update the cholesky decomposition
        C = np.ndarray((X0.shape[0], X0.shape[0]))
        self.cov.cov[0](self.r[0], X0, A=C)
        C += self.g[0] * np.eye(X0.shape[0])
        B = np.ndarray((self.X[0].shape[0], X0.shape[0]))
        self.cov.cov[0](self.r[0], self.X[0], X0, B)
        try:
            self._LA[0] = update_cholesky(self._LA[0], B, C)
        except scipy.linalg.LinAlgError, err:
            print 'Failed!'
            # The Cholesky decomposition failed.
            # The input point is probably too close to an existing
            # one. So, do not add it at all.
            return False
        self._log_det_A[0] += 2. * np.log(
                np.diag(self._LA[0][self.n_of[0]:, self.n_of[0]:])).sum()
        # Update the solution of LA[0] * Hs[0] = H[0]
        self._Hs[0] = update_cholesky_linear_system(self._Hs[0],
                self._LA[0], H0)
        # Update the QR factorization
        self._QLAiH[0], self._RLAiH[0] = scipy.linalg.qr(self._Hs[0],
                mode='full')
        #self._QLAiH[0], self._RLAiH[0] = update_qr(self._QLAiH[0],
        #        self._RLAiH[0], self._Hs[0][self.n_of[0]:, :])
        self._log_det_HTAiH[0] = 2. * np.log(np.abs(
                np.diag(self._RLAiH[0]))).sum()
        # Resize the dependent variables
        self._X = (np.vstack([self._X[0], X0]), ) + self._X[1:]
        self._H = (np.vstack([self._H[0], H0]), ) + self._H[1:]
        self._Y = np.vstack([self._Y, Y0])
        n_new = self.n_of[0] + X0.shape[0]
        self._n_of = (n_new, ) + self.n_of[1:]
        self._n = np.prod(self.n_of)
        self._LAn[0] = np.ndarray((n_new, n_new), order='F')
        self._Hsn[0] = np.ndarray((n_new, self.m_of[0]), order='F')
        self._QLAiHn[0] = np.ndarray((n_new, n_new), order='F')
        self._RLAiHn[0] = np.ndarray((n_new, self.m_of[0]), order='F')
        self._Ys = np.ndarray((self.n, self.q), order='F')
        self._Ysn = np.ndarray((self.n, self.q), order='F')
        self._YmHBs = np.ndarray((self.n, self.q), order='F')
        self._YmHBsn = np.ndarray((self.n, self.q), order='F')
        self._log_det_Sigma = self._compute_dep( self._LA, self._Hs,
                        self._QLAiH, self._RLAiH,
                        self._Ys, self._B, self._YmHBs, self._Sigma,
                        self._LSigma)
        self._log_post_lk, self._log_like = self._compute_log_post_lk()
        return True

    def rg_to_hyp(self):
        """Get a 1D representation of the hyper-parameters."""
        return np.hstack([np.hstack(self.r), self.g])

    def copy(self):
        """Get a copy of the surrogate."""
        surrogate_sample = MultioutputGaussianProcess()
        surrogate_sample.set_data(self.X, self.H, self.Y)
        hyp = self.rg_to_hyp
        surrogate_sample.initialize(hyp, eval_state=self.current_state)
        return surrogate_sample

    def sample_surrogate(self, X_design, H_design,
            rel_tol=0.1, abs_tol=1e-3):
        """Sample a surrogate surface.

        Samples a surrogate surface that can be evaluated analytically. The
        procedure adds the design point with the maximum uncertainty defined
        by Eq. (19) of the paper and assuming a uniform input distribution
        until:
            + we run of design points,
            + or the <global> uncertainty satisfies a stopping criterion.
        The global uncertainty is defined to be the average uncertainty of
        all design points. The stopping criterion is implemented as follows:
            STOP if global uncertainty < rel_tol * init_unc or < abs_tol,
        where init_unc is the initial uncertainty and rel_tol is a relative
        reduction and abs_tol is the absolute uncertainty we are willing to
        accept.

        Arguments:
            X_design        ---     The design points to be used. This
                                    should be as dense as is computationally
                                    feasible.
            rel_tol         ---     We stop if the current uncertainty
                                    is rel_tol times the initial uncertainty.
            abs_tol         ---     We stop if the current uncertainty is
                                    less than abs_tol.
        """
        assert isinstance(X_design, np.ndarray)
        assert X_design.ndim <= 2
        if X_design.ndim == 1:
            X_design = X_design.reshape((X_design.shape[0], 1))
        assert X_design.shape[1] == self.k_of[0]
        assert isinstance(H_design, np.ndarray)
        assert H_design.ndim <= 2
        if H_design.ndim == 1:
            H_design = H_design.reshape((H_design.shape[0], 1))
        assert X_design.shape[0] == H_design.shape[0]
        assert H_design.shape[1] == self.m_of[0]
        assert isinstance(rel_tol, float)
        assert rel_tol >= 0.
        assert isinstance(abs_tol, float)
        assert abs_tol >= 0.
        #1. Create a copy of the current model
        surrogate_sample = self.copy()
        # Do this untill the uncertainty is reduced by a given ammount or
        # X_design is empty
        n = np.prod(self.n_of[1:])
        Y = np.ndarray((n, self.q), order='F')
        Y_max = np.ndarray((n, self.q), order='F')
        C = np.ndarray((n, n), order='F')
        in0it_unc = None
        n_design = X_design.shape[0]
        while X_design.shape[0] > 0:
            total_unc = 0.
            max_unc = 0.
            max_idx = -1
            # Loop over all design points
            for i in xrange(X_design.shape[0]):
                X = (X_design[i, :], ) + self.X[1:]
                H = (H_design[i, :], ) + self.H[1:]
                unc = surrogate_sample.sample_prediction(X, H, Y=Y, C=C)
                total_unc += unc
                if unc > max_unc:
                    max_unc = unc
                    max_idx = i
                    Y_max, Y = Y, Y_max
                #print i, ': unc = ', unc
            total_unc /= n_design
            print 'total_unc: ', total_unc
            if init_unc is None:
                init_unc = total_unc
            if total_unc < rel_tol * init_unc or total_unc < abs_tol:
                print '*** Converged.'
                break
            assert max_idx >= 0
            # Add the best design point to the data set and continue
            surrogate_sample.add_data(X_design[max_idx, :],
                    H_design[max_idx, :], Y_max)
            # Remove the particle we just used.
            X_design = np.delete(X_design,  max_idx, 0)
            H_design = np.delete(H_design, max_idx, 0)
        return surrogate_sample, total_unc

    def evaluate_sparse(self, X, H, compute_covariance=False, sp_tol=0.1):
        """Evaluates the prediction at a given set of points.

        Same as __call__ but we attemp to use sparse matrices.

        The result of this function, is basically the predictive
        distribution, encoded in terms of the mean Y and the covariance
        matrix C.

        Arguments:
            X   ---     The input points.
            H   ---     The design matrices.

        Keyword Arguments:
            compute_covariance  ---     If set to C is None, and the flag
                                        is set to True, then the covariance
                                        is calculated and returned. If C is
                                        not None, then it is ignored.
        """
        if not isinstance(X, tuple) and not isinstance(X, list):
            X = [X]
        if not isinstance(H, tuple) and not isinstance(H, list):
            H = [H]
        if not isinstance(X, list):
            X = list(X)
        if not isinstance(H, list):
            H = list(H)
        for i in xrange(self.s):
            assert X[i].ndim <= 2
            if X[i].ndim == 1:
                X[i] = X[i].reshape((X[i].shape[0], 1))
            assert X[i].shape[1] == self.k_of[i]
            assert H[i].ndim <= 2
            if H[i].ndim == 1:
                H[i] = H[i].reshape((H[i].shape[0], 1))
            assert H[i].shape[1] == self.m_of[i]
        n_of = tuple(x.shape[0] for x in X)
        n = np.prod(n_of)
        Y = np.ndarray((n, self.q), order='F')
        # Compute the mean
        T = ()
        for i in range(self.s):
            tmp = np.ndarray((self.n_of[i], n_of[i]), order='F')
            self.cov.cov[i](self.r[i], self.X[i], X[i], tmp)
            tmp1 = scipy.linalg.solve_triangular(self._LA[i], tmp,
                                                 lower=True)
            T += (tmp1.T, )
        Y[:] = kron_prod(H, self._B)
        tmp1 = kron_prod(T, self._YmHBs)
        Y += tmp1
        if not compute_covariance:
            return Y
        # Compute the covariance
        C0 = ()
        for i in range(self.s):
            C0_tmp = self.cov.cov[i](self.r[i], X[i])
            C0_tmp[C0_tmp < sp_tol] = 0.
            C0 += (sp.csr_matrix(C0_tmp), )
        C = C0[0]
        for i in range(1, self.s):
            C = sp.kron(C, C0[i])
        print 'shape of C: ', C.shape
        print 'non-zeros of C: ', C.nnz
        C1 = T[0]
        for i in range(1, self.s):
            C1 = np.kron(C1, T[i])
        C1[C1 < sp_tol] = 0.
        C1 = sp.csr_matrix(C1)
        C = C - C1 * C1.T
        print 'shape of C: ', C.shape
        print 'non-zeros of C: ', C.nnz
        return Y, C
        tmpm = ()
        for i in range(self.s):
            tmpm += (np.dot(self._Hs[i].T, T[i].T), )
        tmpm_3 = tmpm[0]
        tmpm_4 = H[0]
        for i in range(1, self.s):
            tmpm_3 = np.kron(tmpm_3, tmpm[i])
            tmpm_4 = np.kron(tmpm_4, H[i])
        C2 = tmpm_4.T - tmpm_3
        R = ()
        for m, Rc in iter.izip(self.m_of, self._RLAiH):
            R += (Rc[:m, :m], )
        uplo = 'U' * self.s
        C2[:] = kron_solve(R, C2)
        C2[np.abs(C2) < sp_tol] = 0.
        C2 = sp.csr_matrix(C2)
        #C += np.dot(C2.T, C2)
        Cp = C2.T * C2
        #C = C + C2.T * C2
        print 'shape of C: ', C.shape
        print 'non-zeros of C: ', C.nnz
        quit()
        return Y, C