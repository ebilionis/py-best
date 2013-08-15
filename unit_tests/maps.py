"""Unit-tests for best.maps

Author:
    Ilias Bilionis

"""


if __name__ == '__main__':
    import fix_path


import unittest
import numpy as np
import best.maps


class MapsTest(unittest.TestCase):

    def test_wrapper_function(self):
        return
        def f(x):
            return x + x
        ff = best.maps.Function(10, 10, name='foo', f_wrapped=f)
        print str(ff)
        x = np.random.randn(10)
        print 'Eval at', x
        print ff(x)
        x = np.random.randn(50, 20, 10)
        y = ff(x)
        print y.shape

    def test_sum_function(self):
        def f(x):
            return x + x
        def g(x):
            return x ** 2
        ff = best.maps.Function(10, 10, name='foo', f_wrapped=f)
        gg = best.maps.Function(10, 10, name='foo', f_wrapped=g)
        c = 10.
        x = np.random.randn(10)
        fpg = ff + gg
        print str(fpg)
        print 'Eval at', x
        print fpg(x)
        fpc = ff + c
        print str(fpc)
        print 'Eval at', x
        print fpc(x)

    def test_multiply_function(self):
        def f(x):
            return x + x
        def g(x):
            return x ** 2
        ff = best.maps.Function(10, 10, name='foo', f_wrapped=f)
        gg = best.maps.Function(10, 10, name='foo', f_wrapped=g)
        c = 10.
        x = np.random.randn(10)
        fpg = ff * gg
        print str(fpg)
        print 'Eval at', x
        print fpg(x)
        fpc = ff * c
        print str(fpc)
        print 'Eval at', x
        print fpc(x)

    def test_compose_function(self):
        def f(x):
            return x + x
        def g(x):
            return x ** 2
        ff = best.maps.Function(10, 10, name='foo', f_wrapped=f)
        gg = best.maps.Function(10, 10, name='foo', f_wrapped=g)
        c = 10.
        x = np.random.randn(10)
        h = best.maps.FunctionComposition((ff, gg))
        print str(h)
        print 'Eval at', x
        print h(x)

    def test_power_function(self):
        def f(x):
            return x + x
        ff = best.maps.Function(10, 10, name='foo', f_wrapped=f)
        c = 10.
        x = np.random.randn(10)
        h = best.maps.FunctionPower(ff, 2.)
        print str(h)
        print 'Eval at', x
        print h(x)

    def test_screen_function(self):
        return
        def f(x):
            return x * 2.
        ff = best.maps.Function(10, 10, name='foo', f_wrapped=f)
        h = best.maps.FunctionScreened(ff, in_idx=[0, 4],
                             default_inputs=np.ones(ff.num_input) * .5,
                             out_idx=[3, 5])
        print 'Evaluate h at x = [0.3, -1.]:'
        print h(np.array([0.3, -1.]))
        print 'It should be equivalent to evaluating: '
        x_full = np.ones(ff.num_input) * .5
        x_full[[0, 4]] = np.array([0.3, -1.])
        print f(x_full)[[3, 5]]


if __name__ == '__main__':
    unittest.main()