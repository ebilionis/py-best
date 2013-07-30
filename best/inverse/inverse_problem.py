"""Define the InverseProblem class.

Author:
    Ilias Bilionis

Date:
    1/14/2013
    1/21/2013
"""

import numpy as np
import itertools
from uq.random import StudentTLikelihoodFunction
from uq.random import RandomWalkProposal
from uq.random import SequentialMonteCarlo


class InverseProblem(object):
    """The general inverse problem class."""

    # The SMC object
    _smc = None

    # The final particles
    _r = None

    # The final weights
    _w = None

    # A resampled version of the particles
    _resampled_r = None

    # The mean of the particles
    _mean = None

    # The variance of the particles
    _variance = None

    @property
    def smc(self):
        """Get the SMC object."""
        return self._smc

    @property
    def alpha(self):
        """Get the alpha parameter of the Gamma dist. for the precision."""
        return self._alpha

    @property
    def beta(self):
        """Get the beta parameter of the Gamma dist. for the precision."""
        return self._beta

    @property
    def particles(self):
        """Get the final particles."""
        return self._r

    @property
    def weights(self):
        """Get the final weights."""
        return self._w

    @property
    def resampled_particles(self):
        """Get the resampled particles."""
        return self._resampled_r

    @property
    def mean(self):
        """Get the mean of the particles."""
        return self._mean

    @property
    def variance(self):
        """Get the variance of the particles."""
        return self._variance

    def __init__(self, solver=None, prior=None, data=None, alpha=1e-2, beta=1e-2,
                 verbose=True, mpi=None, comm=None, num_particles=100,
                 num_mcmc=10, proposal=RandomWalkProposal(dt=0.2),
                 store_intermediate_samples=False):
        """Initialize the object.

        Keyword Arguments:
            solver  ---     The forward solver you wish to use.
            prior   ---     The prior distribution of the parameters.
            proposal---     The MCMC proposal.
            alpha   ---     The alpha parameter (shape) of the Gamma
                            distribution of the precision of the forward solver.
            beta    ---     The beta parameter (rate) of the Gamma
                            distribution of the precision of the forward solver.
            verbose ---     Be verbose ir not.
            mpi     ---     Use MPI or not.
            comm    ---     The MPI communicator.
            num_particles ---   The number of particles.
            num_mcmc      ---   The number of MCMC steps per SMC step.
            proposal      ---   The MCMC proposal.
        """
        if solver is None:
            raise ValueError('The forward solver must be specified.')
        if data is None:
            raise ValueError('The data must be specified.')
        if prior is None:
            raise ValueError('The prior must be specified.')
        likelihood = StudentTLikelihoodFunction(2. * alpha, num_input=prior.num_input,
                                                data=data,
                                                mean_function=solver,
                                                cov=(beta / alpha))
        self._smc = SequentialMonteCarlo(prior=prior, likelihood=likelihood,
                                         verbose=verbose, num_particles=num_particles,
                                         num_mcmc=num_mcmc, proposal=proposal,
                                         store_intermediate_samples=store_intermediate_samples,
                                         mpi=mpi, comm=comm)

    def solve(self):
        """Solve the inverse problem."""
        r, w = self.smc.sample()
        self._r = r
        self._w = w
        idf = lambda(x): x
        self._mean = self.mean_of(idf)
        self._variance = self.variance_of(idf, self.mean)
        return r, w

    def mean_of(self, function):
        """Calculate the mean of a function of the particles."""
        y = np.array([self._w[i] * function(self._r[i,:]) for i in
            xrange(self._r.shape[0])])
        return np.mean(y, axis=0)

    def variance_of(self, function, mean=None):
        """Calculate the variance of a function"""
        if mean is None:
            mean = self.mean_of(function)
        v = np.array([self._w[i] * (function(self._r[i, :]) - mean) ** 2
            for i in xrange(self._r.shape[0])])
        return np.mean(v, axis=0)
