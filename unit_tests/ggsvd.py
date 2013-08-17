"""Unit-tests for best.core.ggsvd

Author:
    Ilias Bilionis

Date:
    8/16/2013
"""


if __name__ == '__main__':
    import fix_path


import unittest
import numpy as np
import best.core


class GGSVDTest(unittest.TestCase):

    def test_ggsvd(self):
        A = np.arange(50).reshape((10, 5))
        B = np.arange(15).reshape((3, 5))
        print 'A: '
        print A
        print 'B: '
        print B
        gsvd = best.core.GeneralizedSVD(A, B)
        print gsvd.A.flags
        print gsvd.A
        print 'U: '
        print gsvd.U
        print 'Q: '
        print gsvd.Q
        print 'alpha: '
        print gsvd.alpha
        print gsvd.k
        print gsvd.l
        print gsvd.m
        print gsvd.R
        print gsvd.C
        print gsvd.S
        print gsvd.C ** 2 + gsvd.S ** 2
        print gsvd.D1
        print gsvd.D2
        ZR = np.zeros((gsvd.k + gsvd.l, gsvd.n))
        ZR[:, gsvd.n - gsvd.k - gsvd.l:] = gsvd.R
        print np.dot(gsvd.U, np.dot(gsvd.D1, np.dot(ZR, gsvd.Q.T)))
        print np.dot(gsvd.V, np.dot(gsvd.D2, np.dot(ZR, gsvd.Q.T)))

if __name__ == '__main__':
    unittest.main()