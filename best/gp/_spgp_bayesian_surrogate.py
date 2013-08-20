"""A SPGP Bayesian surrogate.

Author:
    Ilias Bilionis

Date:
    3/13/2013
"""


__all__ = ['SPGPBayesianSurrogate']


import numpy as np
from ._spgp import *
from ..maps import Function
import itertools


class SPGPBayesianSurrogate(Function):
    """A SPGP Bayesian surrogate for one output."""

    # A surrogate for each particle (SPGPSurrogate list)
    _particles = None

    # The weights of the particles (np.ndarray)
    _weights = None

    @property
    def particles(self):
        return self._particles

    @property
    def weights(self):
        return self._weights

    @property
    def num_particles(self):
        return len(self._particles)

    def __init__(self, particles, weights):
        super(SPGPBayesianSurrogate, self).__init__(particles[0].num_input,
                                                    particles[0].num_output,
                                                    name='SPGP Surrogate')
        self._particles = particles
        self._weights = weights

    def __call__(self, x, return_variance=False, add_noise=False):
        mu = np.zeros((x.shape[0], self.num_output))
        s2 = np.zeros((x.shape[0], self.num_output))
        for p, w in itertools.izip(self.particles, self.weights):
            p_mu, p_s2 = p(x, return_variance=True, add_noise=add_noise)
            mu += w * p_mu
            s2 += w * p_s2
        if return_variance:
            return mu, s2
        else:
            return mu

    def _sample_particle(self):
        """Sample a particle."""
        I = np.random.multinomial(1, self.weights)
        return np.arange(self.num_particles)[I == 1][0]

    def sample(self, x, num_samples=1, add_noise=False):
        # Sample a particle
        samples = []
        for i in range(num_samples):
            idx = self._sample_particle()
            samples += self.particles[idx].sample(x, num_samples=1,
                                                  add_noise=add_noise)
        return samples