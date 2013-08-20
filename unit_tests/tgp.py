"""Unit-tests for rvm.

Author:
    Ilias Bilionis

Date:
    8/17/2013
"""


if __name__ == '__main__':
    import fix_path


import unittest
import numpy as np
import scipy.linalg
from best.gp import TreedMultioutputGaussianProcess
import matplotlib.pyplot as plt


class RVMTest(unittest.TestCase):

    def test_gp(self):
        # Number of observations
        num_obs = 20
        # The noise we will add to the data (std)
        noise = 1e-6
        # Draw the observed input points randomly
        X = -10. + 20. * np.random.rand(num_obs)
        X = np.atleast_2d(X).T
        # Draw the observations
        Y = np.sin(X) / (X + 1e-6) + noise * np.random.randn(*X.shape)
        # Construct the design matrix
        H = np.ones(X.shape)
        # Use RVM on the data
        gp = MultioutputGaussianProcess()
        gp.set_data(X, H, Y)
        # Pick the hyper-parameters (length scales, nuggets)
        hyp = np.array([1., 1e-6])
        gp.initialize(hyp)
        # Run 2000 MCMC steps
        gp.sample(steps=2000)
        # Get a function object (subject to change in the future)
        f = gp
        plt.plot(X, Y, '+', markersize=10)
        x = np.linspace(-10, 10, 100)
        x = np.atleast_2d(x).T
        h = np.ones(x.shape)
        fx, Cx = f(x, h, compute_covariance=True)
        plt.plot(x, fx, 'b', linewidth=2)
        plt.plot(x, np.sin(x) / (x + 1e-6), 'r', linewidth=2)
        s2 = 2. * np.sqrt(np.diag(Cx)).reshape(fx.shape)
        plt.plot(x, fx + s2, 'g')
        plt.plot(x, fx - s2, 'g')
        plt.show()


if __name__ == '__main__':
    unittest.main()