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
        #x = -10 + 20. * np.random.rand(50)
        x = np.random.randn(2000)
        #x = np.linspace(-10, 10, 100)
        y = np.sin(x) / (x + 1e-6) + 0.001 * np.random.randn(*x.shape)
        cov = best.maps.CovarianceFunctionSE(1)
        cov.hyp = 2.
        #basis = cov.to_basis(x)
        basis = best.gpc.OrthogonalPolynomial(20, rv=scipy.stats.norm(),
                                              ncap=200)
        #basis = basis.join(basis2)
        phi = basis(x)
        rvm = best.rvm.RelevanceVectorMachine()
        rvm.set_data(phi, y)
        rvm.initialize()
        rvm.train(tol=1e-4, verbose=True)
        f = rvm.get_generalized_linear_model(basis)
        plt.plot(x, y, '+')
        t = np.linspace(-6, 6, 100)
        ft = f(t)
        plt.plot(t, ft, 'b', linewidth=2)
        plt.plot(t, np.sin(t) / (t + 1e-6), 'r', linewidth=2)
        s2 = 2. * np.sqrt(f.get_predictive_variance(t))
        plt.plot(t, ft + s2, 'g')
        plt.plot(t, ft - s2, 'g')
        #plt.plot(x[rvm.relevant], y[rvm.relevant], 'om')
        #plt.plot(t, f.d(t))
        #plt.plot(t, (t * np.cos(t) - np.sin(t)) / (t ** 2 + 1e-6))
        #plt.plot(t, basis(t)[:, rvm.relevant], 'y')
        plt.show()


if __name__ == '__main__':
    unittest.main()