from uq.gp import *
import numpy as np
import matplotlib.pylab as plt


if __name__ == '__main__':
    x = np.loadtxt('train_inputs', ndmin=2)
    y = np.loadtxt('train_outputs', ndmin=2)
    me_y = np.mean(y)
    y0 = y - me_y
    x_test = np.loadtxt('test_inputs', ndmin=2)
    num_pseudo = 20
    prior = SPGPPrior(x, num_pseudo, r_scale=0.1,
                      s_scale=1., g_scale=0.1)
    likelihood = SPGPLikelihood(x, y0, num_pseudo)
    posterior = SPGPPosterior(prior, likelihood)
    proposal = SPGPProposal(dt=[1e-2, 1e-2, 1e-2, 1e-2])
    mcmc = SPGPMCMCGibbs(posterior, proposal, verbose=False, num_gibbs=[1, 1, 1, 1])
    num_particles = 64
    smc = SPGPSMC(mcmc, num_particles=num_particles, num_mcmc=1, verbose=True)
    theta, w = smc.sample()
    s = create_SPGPBayesianSurrogate(x, y0, theta, w, y_mean=me_y)
    plt.clf()
    mu, s2 = s(x_test, return_variance=True)
    plt.plot(x, y, 'm.')
    plt.plot(x_test, mu, 'b')
    plt.plot(x_test, mu + 2. * np.sqrt(s2), 'r')
    plt.plot(x_test, mu - 2. * np.sqrt(s2), 'r')
    plt.show()