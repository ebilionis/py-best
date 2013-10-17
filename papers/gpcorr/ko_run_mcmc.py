"""
Train the GP correlations model for KO with MCMC.
"""

import sys
sys.path.insert(0, '../..')
import numpy as np
from examples.ko import KOSolver
import model
import best
import pymc
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Set the sampling parameters
    k = 1               # Number of dimensions
    n_t = 10            # Number of time steps
    num_samples = 20    # Number of samples

    # Initialize the solver
    solver = KOSolver(k=k, n_t=n_t)

    # Collect data
    X = best.design.latin_center(num_samples, k)
    Y = []
    for i in range(num_samples):
        Y.append(solver(X[i, :]))
    Y = np.vstack(Y)

    cgp_model = model.make_model((X, solver.X_fixed[0]), Y)

    mcmc_sampler = pymc.MCMC(cgp_model)

    mcmc_sampler.sample(100000, thin=1000, burn=10000)

    pymc.Matplot.plot(mcmc_sampler)
    plt.show()
