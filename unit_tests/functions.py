"""Unit-tests for best.gpc

Author:
    Ilias Bilionis

Date:
    8/20/2013

"""


if __name__ == '__main__':
    import fix_path


import unittest
import numpy as np
import math
import scipy.stats as stats
import matplotlib.pyplot as plt

from best.maps import *


class Func1(Function):

    def __init__(self):
        super(Func1, self).__init__(2, 3, num_hyp=1, name='Func1')

    def _eval(self, x, hyp):
        y = np.zeros(self.num_output)
        y[0] = x[0] **2 - x[1] * hyp[0]
        y[1] = hyp[0] * x[1] ** 3
        y[2] = x[0] * x[1]
        return y

    def _d_eval(self, x, hyp):
        y = np.zeros((self.num_output, self.num_input))
        y[0, :] = [2. * x[0], -hyp[0]]
        y[1, :] = [0., 3. * hyp[0] * x[1] ** 2]
        y[2, :] = [x[1], x[0]]
        return y

    def _d_hyp_eval(self, x, hyp):
        y = np.zeros((self.num_output, self.num_hyp))
        y[0, :] = -x[1]
        y[1, :] = x[1] ** 3
        y[2, :] = 0.
        return y


class FunctionsTest(unittest.TestCase):

    def test_function(self):
        return
        f1 = Func1()
        print str(f1)
        print f1([0., 1], hyp=2.)
        print f1.d([0., 1], hyp=2.)
        print f1.d_hyp([[0., 1.], [1., 4]], hyp=2.)

    def test_function_sum(self):
        return
        f1 = Func1()
        f2 = Func1()
        f = f1 + f2
        print str(f)
        print f1([0., 1.], hyp=2.)
        print '+'
        print f2([0., 1.], hyp=1.)
        print '---------------'
        print f([0., 1.], hyp=[2., 1.])
        print f1.d([0., 1.], hyp=2.)
        print '+'
        print f2.d([0., 1.], hyp=1.)
        print '---------------'
        print f.d([0., 1.], hyp=[2., 1.])
        print f1.d_hyp([0., 1.], hyp=2.)
        print '+'
        print f2.d_hyp([0., 1.], hyp=1.)
        print '---------------'
        print f.d_hyp([0., 1.], hyp=[2., 1.])

    def test_function_product(self):
        f1 = Func1()
        f2 = Func1()
        f = f1 * f2
        print str(f)
        print f1([0., 1.], hyp=2.)
        print '*'
        print f2([0., 1.], hyp=1.)
        print '---------------'
        print  f([0., 1.], hyp=[2., 1.])
        print f1.d([0., 1.], hyp=2.)
        #print '+'
        print f2.d([0., 1.], hyp=1.)
        #print '---------------'
        print f.d([0., 1.], hyp=[2., 1.])
        #print f1.d_hyp([0., 1.], hyp=2.)
        #print '+'
        #print f2.d_hyp([0., 1.], hyp=1.)
        #print '---------------'
        #print f.d_hyp([0., 1.], hyp=[2., 1.])
        print f.d([[0., 1.], [1., 3.], [2, 5.], [3., 6]], hyp=[2., 1.])
        print f.d_hyp([0., 1.], hyp=[2., 1])
        print f.d_hyp([[0., 1.], [1., 3.], [2, 5.], [3., 6]], hyp=[2., 1.])



if __name__ == '__main__':
    unittest.main()