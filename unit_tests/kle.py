"""Unit-tests for best.kle

Author:
    Ilias Bilionis

Date:
    8/19/2013

"""


if __name__ == '__main__':
    import fix_path


import unittest
import numpy as np
import math
import best.maps
import best.random
import scipy.stats as stats
import matplotlib.pyplot as plt


class KLETest(unittest.TestCase):

    def test_kle(self):
        x = np.linspace(0, 1, 50)
        k = best.maps.CovarianceFunctionSE(1)
        A = k(x, x, hyp=0.1)
        kle = best.random.KarhunenLoeveExpansion.create_from_covariance_matrix(A)
        print str(kle)
        y = kle.rvs(size=10)
        print y.shape
        print kle.pdf(y)
        plt.plot(x, kle.rvs(size=10).T)
        plt.show()


if __name__ == '__main__':
    unittest.main()