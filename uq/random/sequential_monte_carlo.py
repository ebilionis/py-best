"""A class that samples from a posterior distribution using SMC.

Author:
    Ilias Bilionis

Date:
    1/15/2013

"""

import numpy as np
from scipy.optimize import brentq
import math
import itertools
from uq.random import PosteriorDistribution
from uq.random import RandomWalkProposal
from uq.random import MarkovChainMonteCarlo
#import matplotlib.pyplot as plt


class SequentialMonteCarlo(object):
    """A SMC class to sample from a posterior.

    A posterior distribution is assumed to be of the form:
        p(x | y) \propto p(y | x) p(x).
    The user must provide:
    + The likelihood p(y | x) which is just a function of x. y is implied
    and not used directly.
    + The prior p(x) from which we must be able to sample directly.
    """

    # The number of particles of this CPU
    _my_num_particles = None

    # The number of particles to be used
    _num_particles = None

    # The logarithm of the weights
    _log_w = None

    # The current effective sample size
    _ess = None

    # The number of MCMC steps for each gamma
    _num_mcmc = None

    # The MCMC proposal
    _proposal = None

    # The particles
    _particles = None

    # The acceptance rate observed by each particle
    _acceptance_rate = None

    # The log likelihood of each particle
    _evaluated_state = None

    # The thresshold of the effective sample size (percentage)
    _ess_threshold = None

    # The reduction of the effective sample size per gamma step
    _ess_reduction = None

    # Do you want to adaptively select the MCMC proposal step?
    _adapt_proposal_step = None

    # The lowest allowed acceptance rate
    _lowest_allowed_acceptance_rate = None

    # The highest allowed acceptance rate
    _highest_allowed_acceptance_rate = None

    # The decrease factor of the proposal step
    _proposal_step_decrease_factor = None

    # The increase factor of the proposal step
    _proposal_step_increase_factor = None

    # Be verbose or not
    _verbose = None

    # Store the intermediate samples or not?
    _store_intermediate_samples = None

    # The intermediate samples (a list of dictionaries with keys:
    # 'gamma', 'r' and 'w').
    _intermediate_samples = None

    # The MPI class
    _mpi = None

    # The MPI communicator
    _comm = None

    # The rank of the CPU
    _rank = None

    # The size of the CPU pool
    _size = None

    # The monte carlo sampler
    _mcmc_sampler = None

    # The maximum allowed proposal step
    _max_proposal_dt = None

    @property
    def my_num_particles(self):
        """Get the my number of particles."""
        return self._my_num_particles

    @property
    def num_particles(self):
        """Get the number of particles."""
        return self._num_particles

    @num_particles.setter
    def num_particles(self, value):
        """Set the number of particles."""
        if not isinstance(value, int):
            raise TypeError('The number of particles must be an int.')
        if value <= 0:
            raise ValueError('The number of particles must be positive.')
        self._my_num_particles = value / self.size
        self._num_particles = self.my_num_particles * self.size
        # Allocate memory
        self._allocate_memory()

    @property
    def log_w(self):
        """Get the log of the weights."""
        return self._log_w

    @property
    def ess(self):
        """Get the current effective sample size."""
        return self._ess

    @property
    def num_mcmc(self):
        """Get the number of MCMC steps per gamma."""
        return self._num_mcmc

    @num_mcmc.setter
    def num_mcmc(self, value):
        """Set the number of MCMC steps per gamma."""
        if not isinstance(value, int):
            raise TypeError('The number of MCMC steps must be an integer.')
        if value <= 0:
            raise ValueError('The number of MCMC steps must be positive.')
        self._num_mcmc = value

    @property
    def particles(self):
        """Get the particles.

        Be carefull, they are actually MCMC samplers.
        """
        return self._particles

    @property
    def evaluated_state(self):
        """Get the evaluated state of each particle."""
        return self._evaluated_state

    @property
    def log_like(self):
        """Get the log likelihood of each particle."""
        log_like = np.array([s['log_like'] for s in self.evaluated_state])
        return log_like

    @property
    def ess_threshold(self):
        """Get the threshold of the effective sample size."""
        return self._ess_threshold

    @ess_threshold.setter
    def ess_threshold(self, value):
        """Set the threshold of the effective sample size.

        It must be a number in (0, 1) representing a percentage of the
        total number of particles. If the effective sample size falls
        below this value, then the particles are automatically
        resampled.
        """
        if not isinstance(value, float):
            raise TypeError('The ESS threshold must be a float.')
        if value <= 0. or value >= 1.:
            raise ValueError('The ESS threshold must be in (0, 1).')
        self._ess_threshold = value

    @property
    def ess_reduction(self):
        """Get the reduction of the effective sample size per gamma step."""
        return self._ess_reduction

    @ess_reduction.setter
    def ess_reduction(self, value):
        """Set the reduction of the effective sample size per gamma step.

        It must be a number in (0, 1) representing the desired reduction
        of the effective sample size when we perform a step in gamma.
        The next gamma will be selected adaptively so that the prescribed
        reduction is achieved.
        """
        if not isinstance(value, float):
            raise TypeError('The ESS reduction must be a float.')
        if value <= 0. or value >= 1.:
            raise ValueError('The ESS reduction must be in (0, 1).')
        self._ess_reduction = value

    @property
    def adapt_proposal_step(self):
        """Get the adapt proposal step flag."""
        return self._adapt_proposal_step

    @adapt_proposal_step.setter
    def adapt_proposal_step(self, value):
        """Set the adapt proposal step flag.

        If the adapt proposal step is set to True, the step
        of the MCMC proposal is adaptively set so that it remains
        between self.lowest_allowed_acceptance_rate and
        self.highest_allowed_acceptance_rate.
        """
        if not isinstance(value, bool):
            raise TypeError('The adpat proposal flag must be a boolean.')
        self._adapt_proposal_step = value

    @property
    def lowest_allowed_acceptance_rate(self):
        """Get the lowest allowed acceptance rate."""
        return self._lowest_allowed_acceptance_rate

    @property
    def highest_allowed_acceptance_rate(self):
        """Get the highest allowed acceptnace rate."""
        return self._highest_allowed_acceptance_rate

    @property
    def proposal_step_decrease_factor(self):
        """Get the proposal step decrease factor."""
        return self._proposal_step_decrease_factor

    @proposal_step_decrease_factor.setter
    def proposal_step_decrease_factor(self, value):
        """Set the proposal step decrease factor."""
        if not isinstance(value, float):
            raise TypeError('The proposal step decrease factor must be a float.')
        if value <= 0. or value >= 1.:
            raise ValueError('The proposal step decrease factor must be in (0, 1).')
        self._proposal_step_decrease_factor = value

    @property
    def proposal_step_increase_factor(self):
        """Get the proposal step increase factor."""
        return self._proposal_step_increase_factor

    @proposal_step_increase_factor.setter
    def proposal_step_increase_factor(self, value):
        """Set the proposal step increase factor."""
        if not isinstance(value, float):
            raise TypeError('The proposal step increase factor must be a float.')
        if value <= 1.:
            raise ValueError('The proposal step increaes factor must be greater than one.')
        self._proposal_step_increase_factor = value

    @property
    def verbose(self):
        """Get the verbosity flag."""
        return self._verbose and self.rank == 0

    @verbose.setter
    def verbose(self, value):
        """Set the verbosity flag."""
        if not isinstance(value, bool):
            raise TypeError('The verbosity flag must be a boolean.')
        self._verbose = value

    @property
    def store_intermediate_samples(self):
        """Get the store intermediate samples flag."""
        return self._store_intermediate_samples

    @store_intermediate_samples.setter
    def store_intermediate_samples(self, value):
        """Set the store intermediate samples flag."""
        if not isinstance(value, bool):
            raise TypeError('The store intermediate samples flag must be a boolean.')
        self._store_intermediate_samples = value

    @property
    def intermediate_samples(self):
        """Get the intermediate samples."""
        return self._intermediate_samples

    @property
    def comm(self):
        """Get the MPI communicator."""
        return self._comm

    @property
    def mpi(self):
        """Get access to the MPI class."""
        return self._mpi

    @comm.setter
    def comm(self, value):
        """Set the MPI communicator."""
        self._comm = value
        if self.use_mpi:
            self._rank = self.comm.Get_rank()
            self._size = self.comm.Get_size()
        else:
            self._rank = 0
            self._size = 1

    @property
    def use_mpi(self):
        """Are we using MPI or not?"""
        return self.comm is not None

    @property
    def rank(self):
        """Get the rank of this CPU."""
        return self._rank

    @property
    def size(self):
        """Get the size of the CPU pool."""
        return self._size

    @property
    def mcmc_sampler(self):
        """Get the mcmc sampler."""
        return self._mcmc_sampler

    @mcmc_sampler.setter
    def mcmc_sampler(self, value):
        """Set the mcmc sampler."""
        assert isinstance(value, MarkovChainMonteCarlo)
        self._mcmc_sampler = value

    @property
    def my_acceptance_rate(self):
        """Get the acceptance rate I observed on my particles."""
        return self._acceptance_rate

    @property
    def acceptance_rate(self):
        """Get the global acceptance rate."""
        if self.use_mpi:
            global_acceptance_rate = self.comm.allreduce(
                    self.my_acceptance_rate)
            return global_acceptance_rate / self.size
        else:
            return self.my_acceptance_rate

    def _logsumexp(self, log_x):
        """Perform the log-sum-exp of the weights."""
        my_max_exp = log_x.max()
        if self.use_mpi:
            max_exp = self.comm.allreduce(my_max_exp, op=self.mpi.MAX)
        else:
            max_exp = my_max_exp
        my_sum = np.exp(log_x - max_exp).sum()
        if self.use_mpi:
            all_sum = self.comm.allreduce(my_sum)
        else:
            all_sum = my_sum
        return math.log(all_sum) + max_exp

    def _normalize(self, log_w):
        """Normalize the weights."""
        c = self._logsumexp(log_w)
        return log_w - c

    def _get_ess_at(self, log_w):
        """Calculate the ESS at given the log weights.

        Precondition:
        The weights are assumed to be normalized.
        """
        log_w_all = log_w
        if self.use_mpi:
            log_w_all = np.ndarray(self.num_particles)
            self.comm.Gather([log_w, self.mpi.DOUBLE],
                [log_w_all, self.mpi.DOUBLE])
        if self.rank == 0:
            ess = 1. / math.fsum(np.exp(2. * log_w_all))
        else:
            ess = None
        if self.use_mpi:
            ess = self.comm.bcast(ess)
        return ess

    def _get_log_of_weight_factor_at(self, gamma):
        """Return the log of the weight factor when going to the new gamma."""
        return (gamma - self.mcmc_sampler.target.gamma) * self.log_like

    def _get_unormalized_weights_at(self, gamma):
        """Return the unormalized weights at a given gamma."""
        return self.log_w + self._get_log_of_weight_factor_at(gamma)

    def _get_ess_given_gamma(self, gamma):
        """Calculate the ESS at a given gamma.

        Return:
        The ess and the normalized weights corresponding to that
        gamma.
        """
        log_w = self._get_unormalized_weights_at(gamma)
        log_w_normalized = self._normalize(log_w)
        return self._get_ess_at(log_w_normalized)

    def _resample(self):
        """Resample the particles.

        Precondition:
        The weights are assumed to be normalized.
        """
        if self.verbose:
            print 'Resampling...'
        idx_list = []
        log_w_all = np.ndarray(self.num_particles)
        if self.use_mpi:
            self.comm.Gather([self.log_w, self.mpi.DOUBLE],
                [log_w_all, self.mpi.DOUBLE])
        else:
            log_w_all = self.log_w
        if self.rank == 0:
            births = np.random.multinomial(self.num_particles,
                                           np.exp(log_w_all))
            for i in xrange(self.num_particles):
                idx_list += [i] * births[i]
        if self.rank == 0:
            idx = np.array(idx_list, 'i')
        else:
            idx = np.ndarray(self.num_particles, 'i')
        if self.use_mpi:
            self.comm.Bcast([idx, self.mpi.INT])
            self.comm.barrier()
            old_particles = self._particles
            old_evaluated_state = self._evaluated_state
            self._particles = []
            self._evaluated_state = []
            for i in xrange(self.num_particles):
                to_whom = i / self.my_num_particles
                from_whom = idx[i] / self.my_num_particles
                if from_whom == to_whom and to_whom == self.rank:
                    my_idx = idx[i] % self.my_num_particles
                    self._particles.append(old_particles[my_idx].copy())
                    self._evaluated_state.append(old_evaluated_state[my_idx].copy())
                elif to_whom == self.rank:
                    self._particles.append(self.comm.recv(source=from_whom, tag=i))
                    self._evaluated_state.append(self.comm.recv(source=from_whom,
                        tag=i))
                elif from_whom == self.rank:
                    my_idx = idx[i] % self.my_num_particles
                    self.comm.send(old_particles[my_idx], dest=to_whom, tag=i)
                    self.comm.send(old_evaluated_state[my_idx], dest=to_whom,
                        tag=i)
                self.comm.barrier()
        else:
            self._particles = [self._particles[i].copy() for i in idx]
            self._evaluated_state = [self._evaluated_state[i].copy() for i in
                idx]
        self.log_w.fill(-math.log(self.num_particles))
        self._ess = self.num_particles
        if self.verbose:
            print 'Done!'

    def _allocate_memory(self):
        """Allocates memory.

        Precondition:
        num_particles have been set.
        """
        if self.verbose:
            print 'Allocating memory...'
        # Allocate and initialize the weights
        self._log_w = np.ones(self.my_num_particles) * (-math.log(self.num_particles))
        # Allocate the particles
        self._particles = []
        for i in xrange(self.my_num_particles):
            self._particles.append(
                    np.zeros(self.mcmc_sampler.target.num_input))
        if self.store_intermediate_samples:
            self._intermediate_samples = []
        ac = self.mcmc_sampler.acceptance_rate
        if isinstance(ac, float):
            self._acceptance_rate = np.ndarray(1)
        else:
            self._acceptance_rate = np.ndarray(
                    self.mcmc_sampler.acceptance_rate.shape)
        if self.verbose:
            print 'Done!'

    def _do_adapt_proposal_step(self):
        """Adjust the proposal step."""
        if self.verbose:
            s = 'Adapting proposal step: ' + str(self.mcmc_sampler.proposal.dt)
        a_rate = self.acceptance_rate
        adapted = False
        if a_rate.shape[0] == 1:
            if a_rate < self.lowest_allowed_acceptance_rate:
                adapted = True
                self.mcmc_sampler.proposal.dt *= (
                        self.proposal_step_decrease_factor)
            if a_rate > self.highest_allowed_acceptance_rate:
                adapted = True
                if (self.mcmc_sampler.proposal.dt < self._max_proposal_dt):
                        self.mcmc_sampler.proposal.dt *= (
                                self.proposal_step_increase_factor)
        else:
            decrease_idx = a_rate < self.lowest_allowed_acceptance_rate
            increase_idx = a_rate > self.highest_allowed_acceptance_rate
            adapted = bool(np.sum(decrease_idx) + np.sum(increase_idx))
            self.mcmc_sampler.proposal.dt[decrease_idx] *= (
                    self.proposal_step_decrease_factor)
            idx = np.array(range(self.mcmc_sampler.proposal.dt.shape[0]))
            idx = idx[increase_idx]
            for i in idx:
                if (self.mcmc_sampler.proposal.dt[i] < self._max_proposal_dt):
                    self.mcmc_sampler.proposal.dt[i] *= (
                            self.proposal_step_increase_factor)
        if self.verbose and adapted:
            print s + ' ---> ' + str(self.mcmc_sampler.proposal.dt)

    def _find_next_gamma(self):
        """Find the next gamma."""
        if self.verbose:
            print 'Finding next gamma.'
        # Define the function whoose root we are seeking
        def f(gamma, args):
            ess_gamma = args._get_ess_given_gamma(gamma)
            return ess_gamma - args.ess_reduction * args.ess
        if f(1., self) > 0:
            return 1.
        else:
            # Solve for the optimal gamma using the bisection algorithm
            gamma = brentq(f, self.mcmc_sampler.target.gamma, 1., self)
            if self.use_mpi:
                self.comm.barrier()
            if self.verbose:
                print 'Done.'
            return gamma

    def _do_store_intermediate_samples(self):
        """Stores the current state in a dictionary."""
        current_state = {}
        current_state['gamma'] = self.mcmc_sampler.target.gamma
        r = [p.copy() for p in self.particles]
        current_state['r'] = np.array(r)
        current_state['w'] = np.exp(self.log_w)
        current_state['dt'] = self.mcmc_sampler.proposal.dt
        current_state['ess'] = self.ess
        self._intermediate_samples.append(current_state)

    def set_lowest_and_highest_acceptance_rates(self, lowest_acceptance_rate,
                                                highest_acceptance_rate):
        """Set the lowest and highest acceptance rates."""
        if not isinstance(lowest_acceptance_rate, float):
            raise TypeError('Acceptance rate must be a float.')
        if not isinstance(highest_acceptance_rate, float):
            raise TypeError('Acceptance rate must be a float.')
        if lowest_acceptance_rate >= highest_acceptance_rate:
            raise ValueError('Must hold: lowest ac. rate < highest acc. rate')
        if lowest_acceptance_rate < 0.:
            raise ValueError('Lowest acceptance rate must be non-negative.')
        if highest_acceptance_rate > 1.:
            raise ValueError(('Highest acceptance rate must be less than or'
                              + ' equal to one.'))
        self._lowest_allowed_acceptance_rate = lowest_acceptance_rate
        self._highest_allowed_acceptance_rate = highest_acceptance_rate

    def __init__(self, mcmc_sampler=None, likelihood=None, prior=None,
                 num_particles=10, num_mcmc=10,
                 proposal=RandomWalkProposal(),
                 ess_threshold=0.67,
                 ess_reduction=0.90,
                 adapt_proposal_step=True,
                 lowest_acceptance_rate=0.2,
                 highest_acceptance_rate=0.5,
                 proposal_step_decrease_factor=0.8,
                 proposal_step_increase_factor=1.2,
                 store_intermediate_samples=False,
                 verbose=False,
                 mpi=None,
                 comm=None):
        """Initialize the object.

        Caution:
        The likelihood and the prior MUST be set!

        Keyword Arguments:
        mcmc_sampler    ---     An mcmc_sampler object. If not specified, then
                                you have to specify a "likelihood", a "prior",
                                and a proposal.
        likelihood      ---     The likelihood function.
        prior           ---     The prior distribution.
        num_particles   ---     The number of particles.
        num_mcmc        ---     The number of MCMC steps per gamma.
        proposal        ---     The MCMC proposal distribution.
        ess_threshold   ---     The ESS threshold below which resampling
                                takes place.
        ess_reduction   ---     The ESS reduction that adaptively specifies
                                the next gamma.
        adapt_proposal_step     ---     Adapt or not the proposal step by
                                        monitoring the acceptance rate.
        lowest_acceptance_rate  ---     If the observed acceptance rate ever
                                        falls below this value, then the MCMC
                                        proposal step is decreased.
        highest_acceptance_rate ---     If the observed acceptance rate ever
                                        goes above this value, then the MCMC
                                        proposal step is increased.
        proposal_step_decrease_factor   ---     The factor multiplying the
                                                proposal step in case the
                                                acceptance rate needs to be
                                                increased.
        proposal_step_increase_factor   ---     The factor multiplying the
                                                proposal step in case the
                                                acceptance rate needs to be
                                                decreased.
        store_intermediate_samples      ---     If set to True, then all intermediate
                                                results are stored.
        verbose     ---     Be verbose or not.
        mpi         ---     set the mpi class.
        comm        ---     Set this to the MPI communicator (If you want to use mpi).

        Caution: The likelihood and the prior must be specified together!
        """
        self._mpi = mpi
        if self.mpi is not None and comm is None:
            self.comm = self.mpi.COMM_WORLD
        elif comm is None:
            self.comm = None
        else:
            raise RunTimeError('To use MPI you have to specify the mpi variable.')
        if likelihood is not None and prior is not None:
            if proposal is None:
                raise ValueError('You must specify a proposal.')
            posterior = PosteriorDistribution(likelihood=likelihood,
                    prior=prior)
            self.mcmc_sampler = MarkovChainMonteCarlo(target=posterior,
                    proposal=proposal)
        elif mcmc_sampler is not None:
            if not isinstance(mcmc_sampler.target, PosteriorDistribution):
                raise TypeError('SMC works only if the target is a'
                        + ' PosteriorDistribution class.')
            self.mcmc_sampler = mcmc_sampler
        else:
            raise ValueError('Specify either an mcmc_sampler or '
                    + ' a prior, a likelihood and a proposal.')
        self.store_intermediate_samples = store_intermediate_samples
        self.num_particles = num_particles
        self.num_mcmc = num_mcmc
        self.ess_threshold = ess_threshold
        self.ess_reduction = ess_reduction
        self.set_lowest_and_highest_acceptance_rates(lowest_acceptance_rate,
                                                     highest_acceptance_rate)
        self.proposal_step_decrease_factor = proposal_step_decrease_factor
        self.proposal_step_increase_factor = proposal_step_increase_factor
        self.verbose = verbose
        self.adapt_proposal_step = adapt_proposal_step
        self._max_proposal_dt = 0.5

    def sample(self):
        """Sample the posterior.

        Return:
        It returns a list of particles and their corresponding weights,
        representing the posterior.
        """
        # 0. Initialize gamma to zero
        self.mcmc_sampler.target.gamma = 0.
        # 1. Initialize all states by sampling from the prior
        if self.verbose:
            print 'Sampling priors.'
        for p in self.particles:
            p[:] = self.mcmc_sampler.target.prior.sample()
        self._evaluated_state = []
        for p in self.particles:
            self._evaluated_state.append(
                    self.mcmc_sampler.target(p, report_all=True))
        # 2. Initialize the weights
        self.log_w.fill(-math.log(self.num_particles))
        self._ess = float(self.num_particles)
        # 2.5 Store intermediate results
        if self.store_intermediate_samples:
            self._do_store_intermediate_samples()
        #for p in self.particles:
        #    plt.plot(p[0], p[1], 'r.')
        #plt.pause(0.01)
        # 3. Loop until gamma reaches one.
        while self.mcmc_sampler.target.gamma < 1.:
            # 4. Find the next gamma sto that the ESS decreases by
            # a specific ammount.
            new_gamma = self._find_next_gamma()
            # 5. Make sure you set the new weights
            log_w = self._get_unormalized_weights_at(new_gamma)
            self._log_w = self._normalize(log_w)
            # and, of course, the new ESS
            self._ess = self._get_ess_at(self.log_w)
            # Update the posterior
            self.mcmc_sampler.target.gamma = new_gamma
            for ps in self.evaluated_state:
                ps['gamma'] = new_gamma
                ps['log_p'] = new_gamma * ps['log_like'] + ps['log_prior']
            # 6. Check if resampling is needed
            if self.ess < self.ess_threshold * self.num_particles:
                self._resample()
            # 7. Perform the MCMC steps
            if self.verbose:
                print 'Performing MCMC at gamma = ', self.mcmc_sampler.target.gamma
            self._acceptance_rate.fill(0.)
            for p, ps in itertools.izip(self.particles, self.evaluated_state):
                q, qs = self.mcmc_sampler.sample(p, eval_state=ps, steps=self.num_mcmc,
                        return_eval_state=True)
                p[:] = q
                ps.update(qs)
                print ps
                self._acceptance_rate += self.mcmc_sampler.acceptance_rate
            self._acceptance_rate /= self.my_num_particles
            if self.verbose:
                print 'AC: ', self._acceptance_rate
                print 'Done!'
            # 8. Check if the proposal step need to be adapted.
            if self.adapt_proposal_step:
                self._do_adapt_proposal_step()
            # 8.5 Store intermediate results
            if self.store_intermediate_samples:
                self._do_store_intermediate_samples()
            #plt.clf()
            #for p, log_w in itertools.izip(self.particles, self.log_w):
            #    plt.plot(p[0], p[1], 'bo',
            #             markersize=100.*math.exp(log_w))
            #plt.axis([0, 1, 0, 1])
            #plt.pause(0.01)
        r = np.vstack(self.particles)
        if self.use_mpi:
            r = np.vstack(self.comm.allgather(r))
        w = np.exp(self.log_w)
        if self.use_mpi:
            w = np.hstack(self.comm.allgather(w))
        # Extract the final particles
        return r, w
