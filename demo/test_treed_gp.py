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
    solver = KOSolver(k=2, T=[0, 1], n_t=32)
    tmgp = TreedMultioutputGaussianProcess(solver=solver)
    tmgp.num_xi_init = 10
    tmgp.num_xi_test = 100
    tmgp.num_max = 11
    tmgp.num_elm_max = 10
    tmgp.verbose = True
    tmgp.model.sample_g = True
    tmgp.model.num_mcmc = 1
    tmgp.model.num_init = 100
    init_hyp = [(.1 * np.ones(k), 1e-1) for k in solver.k_of]
    #init_hyp[0][0][0] = 0.21
    #init_hyp[0][0][1] = 0.37
    #init_hyp[1][0][0] = 0.34
    tmgp.init_hyp = init_hyp
    tmgp.num_mcmc = 500
    tmgp.train()
    print str(tmgp.tree)
    #plt.plot(tmgp.tree.model.X[0], np.zeros(tmgp.tree.model.X[0].shape), '+')
    #plt.show()
    fine_solver = KOSolver(k=solver.k_of[0], n_t=50)
    for i in range(10):
        xi = np.random.rand(1, solver.k_of[0])
        #xi = tmgp.tree.X[0][i:(i + 1), :]
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