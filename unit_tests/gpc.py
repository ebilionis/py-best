"""Unit-tests for best.gpc

Author:
    Ilias Bilionis

Date:
    8/10/2013

"""


if __name__ == '__main__':
    import fix_path


import unittest
import numpy as np
import math
import best.gpc
import matplotlib.pyplot as plt


class GpcTest(unittest.TestCase):

    def test_quadrature_rule(self):
        return
        # Testing [-1, 1]
        quad = best.gpc.QuadratureRule(-1, 1, ncap=10)
        print str(quad)
        # Testing [-1, 1]
        quad = best.gpc.QuadratureRule(0, 1, ncap=10)
        print str(quad)
        # Testing [-inf, inf]
        quad = best.gpc.QuadratureRule(-float('inf'), float('inf'), ncap=10)
        print str(quad)
        # Now some tests with non-identity weight functions.
        wf = lambda(x): np.exp(-x)
        # Testing [-1, 1]
        quad = best.gpc.QuadratureRule(-1, 1, wf=wf, ncap=10)
        print str(quad)
        # Testing [-1, 1]
        quad = best.gpc.QuadratureRule(0, 1, ncap=10)
        print str(quad)
        # Testing [-inf, inf]
        quad = best.gpc.QuadratureRule(-float('inf'), float('inf'), ncap=10)
        print str(quad)
        print 'We will integrate some functions'
        f = lambda(x): np.sin(x)
        quad = best.gpc.QuadratureRule(0, math.pi, ncap=50)
        print quad.integrate(f)

    def test_hermite(self):
        return
        wf = lambda(x): 1. / math.sqrt(2. * math.pi) * np.exp(-x ** 2 / 2.)
        infty = float('inf')
        p = best.gpc.OrthogonalPolynomial(10, left=-infty, right=infty,
                                          wf=wf)
        print p._evaluate_square_norms()
        print str(p)
        x = np.linspace(-2., 2., 100)
        plt.plot(x, p(x))
        plt.show()
        plt.plot(x, p.d(x))
        plt.show()

    def test_laguerre(self):
        return
        wf = lambda(x): np.exp(-x)
        infty = float('inf')
        p = best.gpc.OrthogonalPolynomial(10, left=0, right=infty,
                                          wf=wf)
        print p._evaluate_square_norms()
        print str(p)
        import scipy.stats
        rv = scipy.stats.expon()
        p = best.gpc.OrthogonalPolynomial(10, left=0, right=infty,
                                          wf=rv.pdf)
        x = np.linspace(0., 5., 100)
        plt.plot(x, p(x))
        plt.show()
        plt.plot(x, p.d(x))
        plt.show()

    def test_orthogonal_polynomial(self):
        return
        p = best.gpc.OrthogonalPolynomial(3, left=-1, right=1, wf=lambda(x): 0.5, ncap=50)
        print str(p)
        print p._evaluate_square_norms()
        x = np.linspace(-1, 1, 50)
        print p(x)
        print p.d(x)
        plt.plot(x, p(x), x, p.d(x))
        plt.show()

    def test_beta(self):
        import scipy.stats
        a = 0.3
        b = 0.8
        rv = scipy.stats.beta(a, b)
        p = best.gpc.OrthogonalPolynomial(6, left=0, right=1, wf=rv.pdf)
        x = np.linspace(1e-4, 0.99, 100)
        plt.plot(x, p(x))
        plt.show()


if __name__ == '__main__':
    unittest.main()