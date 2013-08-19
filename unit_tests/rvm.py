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
import best.rvm
import best.maps
import best.gpc
import matplotlib.pyplot as plt


class RVMTest(unittest.TestCase):

    def test_rvm(self):
        # Number of observations
        num_obs = 100
        # The noise we will add to the data (std)
        noise = 0.001
        # Draw the observed input points randomly
        X = -10. + 20. * np.random.rand(num_obs)
        # Draw the observations
        Y = np.sin(X) / (X + 1e-6) + noise * np.random.randn(*X.shape)
        # The covariance function
        k = best.maps.CovarianceFunctionSE(1)
        # Construct the basis
        phi = k.to_basis(X, hyp=2.)
        phi = best.gpc.OrthogonalPolynomial(20, left=-10, right=10)
        # Construct the design matrix
        PHI = phi(X)
        # Use RVM on the data
        rvm = best.rvm.RelevanceVectorMachine()
        rvm.set_data(PHI, Y)
        rvm.initialize()
        rvm.train(verbose=True)
        print str(rvm)
        f = rvm.get_generalized_linear_model(phi)
        plt.plot(X, Y, '+')
        x = np.linspace(-10, 10, 100)
        fx = f(x)
        plt.plot(x, fx, 'b', linewidth=2)
        plt.plot(x, np.sin(x) / (x + 1e-6), 'r', linewidth=2)
        s2 = 2. * np.sqrt(f.get_predictive_variance(x))
        plt.plot(x, fx + s2, 'g')
        plt.plot(x, fx - s2, 'g')
        plt.plot(X[rvm.relevant], Y[rvm.relevant], 'om')
        #plt.plot(t, f.d(t))
        #plt.plot(t, (t * np.cos(t) - np.sin(t)) / (t ** 2 + 1e-6))
        #plt.plot(t, basis(t)[:, rvm.relevant], 'y')
        plt.show()


if __name__ == '__main__':
    unittest.main()