"""Test the KOSolver class.

Author:
    Ilias Bilionis

Date:
    12/2/2012

"""


if __name__ == '__main__':
    import fix_path


from examples.ko import KOSolver
from best.gp import TreedMultioutputGaussianProcess
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Initialize the solver
    solver = KOSolver(k=2, T=[0, 1], n_t=32)
    # Initialize the treed GP
    tmgp = TreedMultioutputGaussianProcess(solver=solver)
    tmgp.num_xi_init = 10
    tmgp.num_xi_test = 100
    tmgp.num_max = 100
    tmgp.num_elm_max = 20
    tmgp.verbose = True
    tmgp.model.sample_g = True
    tmgp.model.num_mcmc = 1
    tmgp.model.num_init = 100
    # Initialial hyper-parameters
    init_hyp = np.array([.1, .1, .1, 1e-1, 1e-1])
    tmgp.init_hyp = init_hyp
    tmgp.num_mcmc = 100
    # Train
    tmgp.train()
    # Print the tree
    print str(tmgp.tree)
    # A fine scale solver to test our predictions
    fine_solver = KOSolver(k=solver.k_of[0], n_t=50)
    # Make predictions
    for i in range(10):
        xi = np.random.rand(1, solver.k_of[0])
        X = [xi] + fine_solver.X_fixed
        H = tmgp.mean_model(X)
        n = np.prod([x.shape[0] for x in X])
        Yp = np.ndarray((n, solver.q), order='F')
        Vp = np.ndarray((n, solver.q), order='F')
        tmgp(X, H, Yp, Vp)
        Y = fine_solver(xi[0, :])
        plt.plot(fine_solver.X_fixed[0], Y)
        E = 2. * np.sqrt(Vp)
        for i in range(solver.q):
            plt.errorbar(fine_solver.X_fixed[0], Yp[:, i], yerr=E[:, i])
        plt.show()