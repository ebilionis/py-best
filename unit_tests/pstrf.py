"""Unittest for incomplete cholesky.

Author:
    Ilias Bilionis

Date:
    8/28/2013
"""


if __name__ == '__main__':
    import fix_path


import numpy as np
import scipy.linalg
import unittest
from best.linalg import IncompleteCholesky
from best.maps import CovarianceFunctionSE


class TestIncompleteCholesky(unittest.TestCase):

    def test_ic(self):
        x = np.linspace(-5, 5, 10)
        f = CovarianceFunctionSE(1)
        A = f(x, x, hyp=10.)
        #np.linalg.cholesky(A)
        ic = IncompleteCholesky(A)
        print 'rank: ', ic.rank
        print 'piv: ', ic.piv
        LL = np.dot(ic.L, ic.L.T)
        print np.linalg.norm((LL - np.dot(ic.P.T, np.dot(A, ic.P))))


if __name__ == '__main__':
    unittest.main()