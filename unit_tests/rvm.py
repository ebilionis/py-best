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
        x = -10 + 20. * np.random.rand(50)
        y = np.sin(x) / (x + 1e-6) + 0.001 * np.random.randn(*x.shape)
        rbf = best.maps.RadialBasisFunctionSE(1)
        rbf.hyp = 1.
        basis = rbf.to_basis(x)
        basis = best.gpc.OrthogonalPolynomial(50, left=-10, right=10,
                                              ncap=200)
        phi = basis(x)
        rvm = best.rvm.RelevanceVectorMachine()
        rvm.set_data(phi, y)
        rvm.initialize(100.)
        rvm.train(tol=1e-6, verbose=True)
        print rvm.relevant
        print rvm.alpha
        print rvm.weights
        print scipy.linalg.det(rvm.sigma)
        weights = np.zeros((basis.num_output, 1))
        weights[rvm.relevant, :] = rvm.weights
        f = best.maps.GeneralizedLinearModel(basis, weights)
        plt.plot(x, y, '+')
        t = np.linspace(-10, 10, 100)
        plt.plot(t, f(t))
        plt.plot(t, np.sin(t) / (t + 1e-6))
        plt.plot(t, basis(t)[:, rvm.relevant])
        plt.show()


if __name__ == '__main__':
    unittest.main()