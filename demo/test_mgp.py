"""Test the Multioutput Gaussian Process."""

import numpy as np
import scipy.linalg
import sys
import cPickle
from uq.gp import *


if __name__ == '__main__':
    Xi = np.loadtxt('tests/data/pflow_128_Xi.dat')
    #Xs = np.loadtxt('tests/data/pflow_Xs.dat')
    Xs1 = np.linspace(0, 1, 32).reshape(32, 1)
    Xs2 = Xs1.copy()
    k = 20
    X = (Xi[1:,:k], Xs1, Xs2)
    H = (np.ones((Xi.shape[0]-1,1)), np.ones((Xs1.shape[0], 1)),
            np.ones((Xs1.shape[0], 1)))
    Y = np.loadtxt('tests/data/pflow_128_Y.dat')[:, 2]
    mgp = MultioutputGaussianProcess()
    mgp.set_data(X, H, Y[32 * 32:])
    hyp = ((5. * np.ones(k), 1e-2),
           (0.2 * np.ones(Xs1.shape[1]), 1e-2),
           (0.2 * np.ones(Xs2.shape[1]), 1e-2))
    mgp.sample_g = True
    mgp.gamma[0] = 1000.
    mgp.sigma_r.fill(1e-2)
    mgp.initialize(hyp)
    print mgp.log_post_lk
    for i in xrange(1000):
        mgp.sample()
        print mgp.log_post_lk, mgp.r, mgp.g
    print str(mgp)
    with open('mgp.dat', 'wb') as fd:
        cPickle.dump(mgp, fd)
    with open('mgp.dat', 'rb') as fd:
        mgp_new = cPickle.load(fd)
    print str(mgp_new)
    Xp = (Xi[0:1, :k], Xs1, Xs2)
    Hp = (np.ones((1, 1)), np.ones((Xs1.shape[0], 1)),
            np.ones((Xs2.shape[0], 1)))
    Yp = np.zeros((1 * 32 * 32, 1))
    mgp(Xp, Hp, Yp)
    for i in range(32 * 32):
        print Yp[i, 0], Y[i]
