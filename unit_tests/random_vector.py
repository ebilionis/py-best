"""Unit tests for best.random.

Author:
    Ilias Bilionis

Date:
    8/11/2013
"""


if __name__ == '__main__':
    import fix_path


import unittest
import scipy.stats as stats
import numpy as np
import math
import matplotlib.pyplot as plt
import best
import best.gpc


class RandomTest(unittest.TestCase):

    def test_conditional_rv(self):
        return
        px = stats.expon()
        py = best.random.RandomVariableConditional(px, (1, 2),
                                                   name='Conditioned Exponential')
        print str(py)
        print py.rvs(size=10)
        print py.interval(0.5)
        print py.median()
        print py.mean()
        print py.var()
        print py.std()
        print py.stats()
        print py.moment(10)
        return
        i = (0, 4)
        t = np.linspace(i[0], i[1], 100)
        plt.plot(t, py.pdf(t), t, py.cdf(t), linewidth=2.)
        plt.legend(['PDF', 'CDF'])
        #plt.show()
        py1, py2 = py.split()
        print str(py1)
        print str(py2)
        plt.plot(t, py1.pdf(t), t, py1.cdf(t),
                 t, py2.pdf(t), t, py2.cdf(t), linewidth=2)
        plt.legend(['PDF $y_1$', 'CDF $y_1$', 'PDF $y_2$', 'CDF $y_2$'])
        #plt.show()


    def test_random_vector(self):
        comp = (stats.expon(), stats.beta(0.4, 0.8), stats.norm())
        rv = best.random.RandomVectorIndependent(comp)
        print str(rv)
        x = rv.rvs()
        print 'One sample: ', x
        print 'pdf:', rv.pdf(x)
        x = rv.rvs(size=10)
        print '10 samples: ', x
        print 'pdf: ', rv.pdf(x)
        print rv.mean()
        print rv.var()
        print rv.std()
        print rv.stats()
        # Split it in two:
        rv1, rv2 = rv.split(0)
        print str(rv1)
        x = rv1.rvs(size=5)
        print x
        print rv1.pdf(x)
        print rv2.pdf(x)
        print str(rv2)
        print x
        x = rv2.rvs(size=5)
        print rv2.pdf(x)
        rv3, rv4 = rv1.split(0)
        print str(rv3)
        print str(rv4)
        rv5, rv6 = rv3.split(1)
        print str(rv5)
        print str(rv6)
        rv7, rv8 = rv5.split(2)
        print str(rv7)
        print str(rv8)

    def test_random_vector_poly(self):
        return
        comp = (stats.expon(), stats.beta(0.4, 0.8), stats.norm())
        rv = best.random.RandomVectorIndependent(comp)
        print str(rv)
        prod = best.gpc.ProductBasis(degree=5, rv=rv)
        print str(prod)
        x = rv.rvs(num_samples=10)
        print x
        print prod(x)

    def test_random_vector_independent(self):
        return
        comp = (stats.expon(), stats.beta(0.4, 0.8), stats.norm())
        rv = best.random.RandomVectorIndependent(comp)
        subdomain = [[0.1, 4.], [0.1, 0.8], [-1., 1.]]
        rvc = best.random.RandomVectorConditional(rv, subdomain)
        print str(rvc)
        return


if __name__ == '__main__':
    unittest.main()