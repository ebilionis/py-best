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
import mpi4py.MPI as mpi
import cPickle as pickle


if __name__ == '__main__':
    # Set the sampling parameters
    k = 1               # Number of dimensions
    n_t = 20            # Number of time steps
    num_samples = 20    # Number of samples
    num_particles = 100
    num_mcmc = 1

    # Initialize the solver
    solver = KOSolver(k=k, n_t=n_t)

    # Collect data
    x = best.design.latin_center(num_samples, k)
    Y = []
    for i in range(num_samples):
        Y.append(solver(x[i, :]))
    X = (x, solver.X_fixed[0])
    H = tuple([np.ones((x.shape[0], 1)) for x in X])
    Y = np.vstack(Y)
    # Save the data for later use
    data_file = 'ko_data_s=%d.pickle' % num_samples
    with open(data_file, 'wb') as fd:
        pickle.dump((X, H, Y), fd, pickle.HIGHEST_PROTOCOL)

    cgp_model = model.make_model(X, Y)

    smc_sampler = best.smc.SMC(cgp_model, num_particles=num_particles,
                                num_mcmc=num_mcmc, verbose=1,
                                gamma_is_an_exponent=True,
                                mpi=mpi)

    smc_sampler.initialize(0.)
    pa = smc_sampler.get_particle_approximation()
    smc_sampler.move_to(1.)
    pa = smc_sampler.get_particle_approximation()
    # dump the result to a file to process later
    gpa = pa.allgather()
    if mpi.COMM_WORLD.Get_rank() == 0:
        out_file = 'ko_smc_s=%d_p=%d_m=%d.pickle' %(num_samples, num_particles,
                                                    num_mcmc)
        with open(out_file, 'wb') as fd:
            pickle.dump(gpa, fd, pickle.HIGHEST_PROTOCOL)
