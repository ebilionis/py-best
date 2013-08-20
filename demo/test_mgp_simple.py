"""A simple test for MGP.

Author:
    Ilias Bilionis

Date:
    11/28/2012

"""


if __name__ == '__main__':
    import fix_path


import numpy as np
from math import *
from best.gp import *


def f(x):
    """A function to learn."""
    if x.shape[0] == 1:
        #return 5 * x[0]
        return sin(10 * x[0])
    else:
        return sin(5 * x[0]) * cos(5 * x[1])


if __name__ == '__main__':
    n = 100
    k = 2
    q = 1
    sigma = 0.
    X = np.random.rand(n, k)
    print X.shape
    H = np.ones((n, 1))
    Y = np.zeros((n, q))
    for i in xrange(n):
        Y[i] = f(X[i, :])
    # Add some noise to the data
    Y += sigma * np.random.randn(n, q)
    mgp = MultioutputGaussianProcess()
    mgp.set_data(X, H, Y)
    hyp = np.array([0.1, 0.1, 1e-2])
    #mgp.num_init = 1000
    mgp.initialize(hyp)
    mgp.sample_g = True
    for i in xrange(5000):
        mgp.sample()
        print i, mgp.log_post_lk
    ns = 100
    #Xp = np.linspace(0, 1, ns).reshape((ns, 1))
    Xp = [np.random.rand(ns, k)]
    Hp = [np.ones((ns, 1))]
    Yp = np.zeros((ns, 1))
    Cp = np.zeros((ns, ns), order='F')
    # mgp(Xp, Hp, Yp)
    Yc = Y.copy()
    mgp(Xp, Hp, Yp, Cp)
    for i in range(ns):
        err = 2. * sqrt(mgp.Sigma[0, 0] * Cp[i, i])
        print Xp[0][i, :], Yp[i], f(Xp[0][i, :]), err