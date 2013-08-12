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
import best
import best.gpc


class RandomTest(unittest.TestCase):

    def test_random_vector(self):
        return
        comp = (stats.expon(), stats.beta(0.4, 0.8), stats.norm())
        rv = best.random.RandomVectorIndependent(comp)
        print str(rv)
        x = rv.rvs(num_samples=10)
        print x
        print rv.pdf(x)
        print rv.logcdf([100., 1., 100.])
        print rv[0].pdf(0.5)

    def test_random_vector_poly(self):
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