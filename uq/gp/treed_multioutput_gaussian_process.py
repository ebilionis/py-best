"""A class that defines a treed Mutioutput Gaussian Process.

Author:
    Ilias Bilionis

Date:
    12/2/2012

"""

import math
import numpy as np
import itertools as iter
from uq import RandomElement
from uq.gp import MultioutputGaussianProcess
from uq.gp import MeanModel
from uq.gp import ConstantMeanModel
from uq.gp import SeparableMeanModel
from uq import Solver
#from uq.random import lhs


class TreedMultioutputGaussianProcess(object):
    """Define a class that describes a treed MGP."""

    # The solver we are going to use
    _solver = None

    # The model we are going to use
    _model = None

    # The mean model
    _mean_model = None

    # The tree
    _tree = None

    # Number of initial traing points
    _num_xi_init = None

    # Number of test points
    _num_xi_test = None

    # Is the model initialized or not
    _is_initialized = None

    # Verbose or not
    _verbose = None

    # The number of mcmc steps used to train each element
    _num_mcmc = None

    # Initialial hyper-parameters
    _init_hyp = None

    @property
    def solver(self):
        """Get the solver."""
        return self._solver

    @solver.setter
    def solver(self, value):
        """Set the solver."""
        if not isinstance(value, Solver):
            raise TypeError('You must use a Solver object.')
        self._solver = value

    @property
    def model(self):
        """Get the model."""
        return self._model

    @model.setter
    def model(self, value):
        """Set the model."""
        if not isinstance(value, MultioutputGaussianProcess):
            raise TypeError('The model must be a Gaussian Process.')
        self._model = value

    @property
    def mean_model(self):
        """Get the mean model."""
        return self._mean_model

    @mean_model.setter
    def mean_model(self, value):
        """Set the mean model."""
        if not isinstance(value, MeanModel):
            raise TypeError('The mean_model must be a MeanModel.')
        self._mean_model = value

    @property
    def tree(self):
        """Get the tree."""
        return self._tree

    @tree.setter
    def tree(self, value):
        """Set the root of the tree."""
        if not isinstance(value, RandomElement):
            raise TypeError('The tree must consist of RandomElements.')
        self._tree = value

    @property
    def num_xi_init(self):
        """Get the number of initial training points."""
        return self._num_xi_init

    @num_xi_init.setter
    def num_xi_init(self, value):
        """Set the number of initial training points."""
        if not isinstance(value, int):
            raise TypeError('The number of initial points must be an int.')
        self._num_xi_init = value

    @property
    def num_xi_test(self):
        """Get the number of ALM points."""
        return self._num_xi_test

    @num_xi_test.setter
    def num_xi_test(self, value):
        """Set the number of ALM points."""
        if not isinstance(value, int):
            raise TypeError('The number of ALM points must be an int.')
        self._num_xi_test = value

    @property
    def is_initialized(self):
        """Is the model initialized?"""
        return self._is_initialized
    
    @property
    def verbose(self):
        """Should I be verbose or not?"""
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        """Set verbose."""
        if not isinstance(value, bool):
            raise TypeError('verbose should be a boolean.')
        self._verbose = value

    @property
    def num_mcmc(self):
        """Get the number of mcmc steps."""
        return self._num_mcmc

    @num_mcmc.setter
    def num_mcmc(self, value):
        """Set the number of mcmc steps."""
        if not isinstance(value, int):
            raise TypeError('num_mcmc must be an int.')
        if not value > 0:
            raise ValueError('num_mcmc must be > 0.')
        self._num_mcmc = value

    @property
    def init_hyp(self):
        """Get the initial hyper-paremeters."""
        return self._init_hyp

    @init_hyp.setter
    def init_hyp(self, value):
        """Set the initial hyper-parameters."""
        self._init_hyp = value

    def __init__(self, solver=Solver,
                 model=MultioutputGaussianProcess(), 
                 mean_model=None,
                 tree=RandomElement(scale_X=True)):
        """Initialize the object.

        Keyword Arguments:
            solver  ---     The solver object you wish to learn.

        """
        self.solver = solver
        self.model = model
        if mean_model is None:
            mean = [ConstantMeanModel(k) for k in self.solver.k_of]
            mean_model = SeparableMeanModel(mean)
        self.mean_model = mean_model
        self.tree = tree
        self.verbose = False
        self.num_mcmc = 100
        self.num_xi_init = 20
        self.num_xi_test = 100

    def initialize(self):
        """Initialize the model."""
        if self.verbose:
            print 'Gathering initial data...'
        Xi_init = lhs(self.num_xi_init, self.solver.k_of[0])
        #x = np.linspace(0, 1, int(math.sqrt(self.num_xi_init)))
        #X1, X2 = np.meshgrid(x, x)
        #Xi_init = np.hstack([X1.reshape((x.shape[0] ** 2, 1)),
        #                     X2.reshape((x.shape[0] ** 2, 1))])
        #Xi_init = np.random.rand(self.num_xi_init, self.solver.k_of[0])
        Y = np.ndarray((self.num_xi_init, np.prod(self.solver.n_of_fixed),
                        self.solver.q), order='F')
        for i in xrange(self.num_xi_init):
            if self.verbose:
                print i + 1, ' out of ', self.num_xi_init
            self.solver(Xi_init[i, :], Y=Y[i, :, :])
            #Y += (0.1 ** 2) * np.random.randn(*Y[i, :, :].shape)
        n = np.prod(Y.shape[:-1])
        Y = Y.reshape((n, self.solver.q))
        X = [Xi_init] + self.solver.X_fixed
        H = self.mean_model(X)
        self.tree.set_data(X, H, Y)

    def _train_element(self, elm, num_mcmc=1000):
        """Train a single element."""
        if self.verbose:
            print 'Training element:'
            print str(elm)
        if not hasattr(elm, 'model'):
            elm.model = MultioutputGaussianProcess(self.model)
            if elm.has_parent and hasattr(elm.parent, 'model'):
                hyp = ()
                for r, g in iter.izip(elm.parent.model.r, elm.parent.model.g):
                    hyp += ((r, g), )
            else:
                hyp = self.init_hyp
            elm.model.set_data(elm.scaled_X, elm.H, elm.Y)
            elm.model.initialize(hyp)
            #elm.model.initialize()
        for i in xrange(num_mcmc):
            elm.model.sample()
            if self.verbose:
                print i+1, elm.model.log_post_lk, elm.model.r, elm.model.g

    def _train_all_elements(self):
        """Train all elements."""
        for elm in self.tree.leaves:
            self._train_element(elm, self.num_mcmc)

    def _refine_element(self, elm):
        """Refine elements."""
        min_c = -1
        min_dim = -1
        m = 1e99
        for c in range(self.solver.s):
            for d in range(self.solver.k_of[c]):
                if elm.model.r[c][d] < m:
                    m = elm.model.r[c][d]
                    min_c = c
                    min_dim = d
        if elm.split(min_c, min_dim):
            if self.verbose:
                print 'Refining element.'
                print 'split_comp: ', min_c
                print 'split_dim: ', min_dim
            return elm.left, elm.right
        else:
            return (elm, )

    def _add_more_points(self):
        """Adds one more point using the ALM criterion.
        
        Return a list of the elements that were updated.
        """
        if self.verbose:
            print 'Active Learning.'
        Xi_test = lhs(self.num_xi_test, self.solver.k_of[0])
        #Xi_test = np.random.rand(self.num_xi_test, self.solver.k_of[0])
        unc_tot = 0.
        unc_max = -1.
        idx = -1
        for i in xrange(self.num_xi_test):
            unc = self.tree.get_uncertainty(Xi_test[i:(i + 1), :])
            unc_tot += unc
            if unc > unc_max:
                unc_max = unc
                idx = i
        unc = unc_max / self.num_xi_test
        X = [Xi_test[idx:(idx + 1), :]] + self.solver.X_fixed
        H = self.mean_model(X)
        Y = self.solver(Xi_test[idx, :])
        return self.tree.add_data(X, H, Y)

    def train(self):
        """Train the model to the solver."""
        if not self.is_initialized:
            self.initialize()
        self._train_all_elements()
        n = self.num_xi_init
        while n < self.num_max:
            updated = self._add_more_points()
            for elm in updated:
                if elm.n_of[0] > self.num_elm_max:
                    for child in self._refine_element(elm):
                        self._train_element(child)
                else:
                    self._train_element(elm, 500)
            n += 1

    def __call__(self, X, H, Y, V=None):
        """Evaluate the model at a particular point."""
        self.tree(X, H, Y, V)
