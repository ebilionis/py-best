"""A general MCMC sampler.

Author:
    Ilias Bilionis

Date:
    1/15/2013

"""

import numpy as np
import math
from uq.random import RandomWalkProposal
from uq.random import LikelihoodFunction
from uq.random import PosteriorDistribution
from uq.random import ProposalDistribution


class MarkovChainMonteCarlo(object):
    """A general MCMC sampler.

    The state of MCMC.
    We assume that the state x of MCMC is a class that implements
    the following methods:
    + x.copy():     Create a copy of x and returns a reference to it.

    The proposal of MCMC.
    The proposal of MCMC should implement the following methods:
    + proposal.__call__(x_p, x_n):      Evaluate the log of the pdf of
                                        p(x_n | x_p) and return the result.
                                        For reasons of computational efficiency,
                                        we had to make the return value of this
                                        function a little bit more complicated than
                                        is necessary in MCMC. We assume that it
                                        returns an dictionary obj that has at least one
                                        field:
                                        + obj['log_p']: The logarithm of the pdf at x.
                                        This object corresponding to the current state
                                        is always stored.
                                        To understand, why something this awkward is
                                        essential, please see the code of the
                                        Sequential MC class.
    + proposal.sample(x_p, x_n):        Sample p(x_n | x_p) and writes the
                                        result on x_n.

    The target distribution.
    The target distribution should simply implement:
    + target.__call__(x):               Evaluate the log of the target pdf up
                                        to a normalizing constant.
    """

    # The target distribution
    _target = None

    # The proposal distribution
    _proposal = None

    # Is the chain initialized?
    _initialized = None

    # The current state of the MCMC
    _current_state = None

    # The proposed state of MCMC
    _proposed_state = None

    # The result of the evaluation of the current state
    _eval_current_state = None

    # Counter of samples taken
    _num_samples = None

    # Counter of samples accepted
    _num_accepted = None

    # Store the samples gathered or not
    _store_samples = None

    # The stored samples
    _samples = None

    # Be verbose or not
    _verbose = None

    # The output frequency
    _output_frequency = None

    @property
    def target(self):
        """Get the target distribution."""
        return self._target

    @target.setter
    def target(self, value):
        """Set the target distribution.

        Everytime the target changes, the chain must be initialized again.
        If the current state is already present, then this method automatically
        reinitializes the chain.

        The target distribution is any class that implements:
        + target.__call__(x):   Evalute the logarithm of the pdf at x, up to a
                                normalizing constant.
        """
        assert isinstance(value, LikelihoodFunction)
        self._target = value
        if self.current_state is not None:
            self.reinitialize()
        else:
            self._initialized = False

    @property
    def proposal(self):
        """Get the proposal distribution."""
        return self._proposal

    @proposal.setter
    def proposal(self, value):
        """Set the proposal distribution.

        The proposal distribution is any class that implements:
        + proposal.__call__(x_p, x_n):  Evaluate the logarithm of the pdf at x_n
                                        given x_p.
        + proposal.sample(x_p, x_n):    Sample x_n given x_p.
        """
        assert isinstance(value, ProposalDistribution)
        self._proposal = value

    @property
    def current_state(self):
        """Get the current state of MCMC."""
        return self._current_state

    @property
    def proposed_state(self):
        """Get the proposed state of MCMC."""
        return self._proposed_state

    @property
    def log_p_current_state(self):
        """Get the log of the pdf of the current state."""
        return self.eval_current_state['log_p']

    @property
    def eval_current_state(self):
        """Get the result of the evaluation of the current state."""
        return self._eval_current_state

    @property
    def num_samples(self):
        """Get the number of samples taken so far."""
        return self._num_samples

    @property
    def num_accepeted(self):
        """Get the number of samples accepted so far."""
        return self._num_accepted

    @property
    def initialized(self):
        """Check if the chain has been initialized."""
        return self._initialized

    @property
    def acceptance_rate(self):
        """Get the acceptance rate."""
        if self.num_samples is None or self.num_samples == 0:
            return 0.
        else:
            return float(self.num_accepeted) / float(self.num_samples)

    @property
    def store_samples(self):
        """Check whether or not the samples are being stored."""
        return self._store_samples

    @store_samples.setter
    def store_samples(self, value):
        """Set the store_sampels flag."""
        if not isinstance(value, bool):
            raise TypeError('The store_samples flag must be a boolean.')
        self._store_samples = value

    @property
    def samples(self):
        """Get the stored samples."""
        return self._samples

    @property
    def verbose(self):
        """Get the verbosity flag."""
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        """Set the verbosity flag."""
        if not isinstance(value, bool):
            raise TypeError('The verbosity flag must be a boolean.')
        self._verbose = value

    @property
    def output_frequency(self):
        """Get the output frequency."""
        return self._output_frequency

    @output_frequency.setter
    def output_frequency(self, value):
        """Set the output frequency."""
        if not isinstance(value, int):
            raise TypeError('The output frequency must be an integer.')
        if value <= 0:
            raise ValueError('The output frequency must be positive.')
        self._output_frequency = value

    def __init__(self, target=None, proposal=RandomWalkProposal(),
                 store_samples=False, verbose=False,
                 output_frequency=1):
        """Initialize the object.

        Keyword Arguments:
        target         ---     The target distribution.
        proposal       ---     The proposal distribution.
        store_samples  ---     If set to True, then all samples are stored
                               (copied) and are accessible via self.samples.
        verbose        ---     The verbosity flag. If set to True, then sth
                               is printed at each MCMC step.
        output_frequency ---   If verbose is True, then this specifies how often
                               sth is printed.
        """
        self.target = target
        self.proposal = proposal
        self.store_samples = store_samples
        self.verbose = verbose
        self.output_frequency = output_frequency

    def _evaluate_target(self, x):
        if isinstance(self.target, PosteriorDistribution):
            return self.target(x, report_all=True)
        else:
            return self.target(x)

    def initialize(self, x, eval_state=None):
        """Initialize the chain.

        Initializes the chain at x. It is essential that the chain has been
        properly initialized!
        """
        if self.verbose:
            print 'Initializing the chain.'
        self._current_state = x
        self._proposed_state = x.copy()
        if eval_state is None:
            self._eval_current_state = self._evaluate_target(x)
        else:
            self._eval_current_state = eval_state.copy()
        self._initialized = True
        self._num_samples = 0
        self._num_accepted = 0
        if self.store_samples:
            self._samples = [x.copy()]

    def reinitialize(self):
        """Re-initializes the chain."""
        self.initialize(self.current_state)

    def _get_log_p_from_eval_state(self, eval_state):
        if isinstance(self.target, PosteriorDistribution):
            return eval_state['log_p']
        else:
            return eval_state

    def perform_single_mcmc_step(self):
        """Performs a single MCMC step.

        The current state of the chain is altered at the end (or not).
        """
        # 1. Propose a move
        self.proposal.sample(self.current_state, self.proposed_state)
        # 2. Evaluate the log of the pdf of the new state
        eval_proposed_state = self._evaluate_target(self.proposed_state)
        log_p_proposed_state = (
                self._get_log_p_from_eval_state(eval_proposed_state))
        # 3. Evaluate log a1 = log p(x_n) - log p(x_p)
        log_a1 = log_p_proposed_state - self.log_p_current_state
        # 4. Evaluate the transition probabilities
        log_p_p_to_n = self.proposal(self.current_state, self.proposed_state)
        log_p_n_to_p = self.proposal(self.proposed_state, self.current_state)
        # 5. Evaluate log a2 = log p(x_p | x_n) - log p(x_n | x_p)
        log_a2 = log_p_n_to_p - log_p_p_to_n
        # 6. Evaluate the acceptance ratio
        log_a = log_a1 + log_a2
        print ' ar:', math.exp(log_a)
        if log_a > 0.:
            a = 1.
        else:
            a = math.exp(log_a)
        # 7. Perform the accept/reject move
        u = np.random.rand()
        if u < a:
            # Accept - swap proposed and current
            self._current_state, self._proposed_state = (self._proposed_state,
                                                         self._current_state)
            self._eval_current_state = eval_proposed_state
            # Increase the number of accepted samples
            self._num_accepted += 1
        # 8. Increase the number of samples
        self._num_samples += 1
        # 9. Add the current state to the stored samples
        if self.store_samples:
            self._samples.append(self.current_state.copy())
        # 10. Print something if requested
        if self.verbose and self.num_samples % self.output_frequency == 0:
            s = 'step = {0:10d}, acc. rate = {1:.2%}'.format(self.num_samples,
                                                         self.acceptance_rate)
            s += ', c. ratio = {0:1.2f}'.format(a)
            s += ', c. log p = {0:10.6f}'.format(self.log_p_current_state)
            print s

    def sample(self, x=None, eval_state=None, return_eval_state=False, steps=1):
        """Sample the chain.

        Keyword Arguments:
        x       ---     The initial state. If not specified, then
                        we assume that it has already been set.
        steps   ---     The number of MCMC steps to be performed.
        return_eval_state   ---     Return the evaluated state at the end of
                                    the run.

        Return:
        A reference to the current state of the chain.
        """
        if x is not None:
            self.initialize(x, eval_state=eval_state)
        for i in xrange(steps):
            #print i, self.current_state, self.eval_current_state
            self.perform_single_mcmc_step()
        if return_eval_state:
            return self.current_state, self.eval_current_state
        else:
            return self.current_state

    def copy(self):
        """Return a copy of this object."""
        copy = MarkovChainMonteCarlo(target=self.target, proposal=self.proposal,
                                     store_samples=self.store_samples,
                                     verbose=self.verbose,
                                     output_frequency=self.output_frequency)
        copy._current_state = self.current_state.copy()
        copy._initialized = self.initialized
        copy._proposed_state = self.proposed_state.copy()
        copy._num_samples = self.num_samples
        copy._num_accepted = self.num_accepeted
        if self.samples is not None:
            copy._samples = self.samples[:]
        copy._eval_current_state = self.eval_current_state.copy()
        return copy
