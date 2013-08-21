"""Test the Sequential Monte Carlo code.

Author:
    Ilias Bilionis

Date:
    1/16/2013

"""


if __name__ == '__main__':
    import fix_path


import numpy as np
import math
from best.random import *
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Number of inputs
    num_input = 1
    # Construct the likelihood function
    # Number of components
    num_comp = 4
    # Degrees of freedom of the Inverse Wishart distribution
    # from which we draw the covariance matrix
    n_d = 10
    # Randomly pick each component
    components = []
    for i in range(num_comp):
        mu = 5. * np.random.randn(num_input)
        X = np.random.randn(n_d, num_input)
        A = np.dot(X.T, X)
        C = np.linalg.inv(A)
        components.append(NormalDistribution(num_input, mu, C))
    # Randomly pick weights for the components
    #w = np.random.rand(num_comp)
    w = np.ones(num_comp) / num_comp
    # Construct the likelihood
    likelihood = MixtureOfDistributions(w, components)
    # Let's just take a look at this distribution
    print 'weights:, ', likelihood.weights
    print 'components:'
    for c in likelihood.components:
        print 'mu: ', c.mu
        print 'cov: ', c.cov
    x = np.linspace(-10., 10., 100.)
    # The prior is just going to be a normal distribution with
    # zero mean and very big variance
    prior = NormalDistribution(num_input, cov=2.)
    # Construct the SMC object
    smc = SequentialMonteCarlo(prior=prior, likelihood=likelihood,
                               verbose=True, num_particles=1000,
                               num_mcmc=10,
                               proposal=RandomWalkProposal(dt=2.),
                               store_intermediate_samples=True)
    r, w = smc.sample()
    step = 0
    for s in smc.intermediate_samples:
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.hist(s['r'], bins=20, weights=s['w'], normed=True)
        ax1.set_xlim([-5., 5.])
        ax1.set_title('gamma = %1.4f' % s['gamma'])
        ax1.set_xlabel('x')
        ax1.set_ylabel('normalized histogram')
        ax2 = fig.add_subplot(1, 2, 2)
        smc.mcmc_sampler.target.gamma = s['gamma']
        log_post = np.array([smc.mcmc_sampler.target(np.array([t])) for t in x])
        ax2.plot(x, np.exp(np.exp(log_post)))
        ax2.set_title('gamma = %1.4f' % s['gamma'])
        ax2.set_xlabel('x')
        ax2.set_ylabel('pdf')
        plt.savefig('smc_step=%d.png' % step)
        step += 1