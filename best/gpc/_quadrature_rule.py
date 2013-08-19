"""
Implements a generic quadrature rule.

Author:
    Ilias Bilionis

Date:
    7/25/2013
"""


__all__ = ['QuadratureRule']


import numpy as np
import math
from .. import Object


def symtr(t):
    """Implements a tranformation of [-1, 1] to [-Infinity, Infinity].

    Return:
        phi(t)  ---     The transformation.
        dphi(t) ---     The derivative of the tranformation.
    """
    t2 = t * t
    dphi = 1. - t2
    phi = t / dphi
    dphi *= dphi
    dphi = (t2 + 1.) / dphi
    return phi, dphi


def tr(t):
    """Implements a transformation of [-1, 1] to [0, Infinity].

    Return:
        phi(t)  ---     The transformation.
        dphi(t) ---     The derivative of the tranformation.
    """
    dphi = 1. - t
    phi = (1. + t) / dphi
    dphi *= dphi
    dphi = 2. / dphi
    return phi, dphi


def fejer(n):
    """Generate the n-point Fejer quadrature rule."""
    x = np.zeros(n)
    w = np.zeros(n)
    nh = n / 2
    np1h = (n + 1) / 2
    fn = float(n)
    for k in range(1, nh + 1):
        x[k - 1] = -math.cos(0.5 * (2. * k - 1.) * math.pi / fn)
        x[n - k] = -x[k - 1]
    if (2 * nh) != n:
        x[np1h - 1] = 0.
    for k in range(1, np1h + 1):
        c1 = 1.
        c0  = 2. * x[k - 1] * x[k - 1] - 1.
        t = 2. * c0
        s = c0 / 3.
        for m in range(2, nh + 1):
            c2 = c1
            c1 = c0
            c0 = t * c1 - c2
            s += c0 / (4. * m * m - 1)
        w[k - 1] = 2. * (1. - 2. * s) / fn
        w[n - k] = w[k - 1]
    return x, w


class QuadratureRule(Object):

    """An object representing a quadrature rule."""

    # The quadrature points (N x D)
    _x = None

    # The quadrature weights (N x 1)
    _w = None

    @property
    def x(self):
        return self._x

    @property
    def w(self):
        return self._w

    @property
    def num_quad(self):
        return self._x.shape[0]

    def __init__(self, left=-1, right=1, wf=lambda(x): 1., ncap=500,
                 name='Quadrature Rule'):
        """Construct a quadrature rule.

        Keyword Arguments
            left    ---     The left end of the interval.
            right   ---     The right end of the interval.
            wf      ---     The weight function. The default is the identity.
            ncap    ---     The number of quadrature points.
            name    ---     A name for the object.
        """
        x, w = fejer(ncap)
        if wf is None:
            wf = lambda(x): np.ones(x.shape)
        if math.isinf(left) and math.isinf(right):
            phi, dphi = symtr(x)
            self._x = phi
        elif math.isinf(right):
            phi, dphi = tr(x)
            self._x = left + phi
        elif math.isinf(left):
            phi, dphi = tr(-x)
            self._x = right - phi
        else:
            self._x = 0.5 * ((right - left) * x + right + left)
            dphi = 0.5 * (right - left)
        self._w = w * wf(self.x) * dphi
        super(QuadratureRule, self).__init__(name=name)

    def integrate(self, f):
        """Integrate the function f.

        When evaluating f(x) with x an N x D matrix,
        then f(x) should be an N x Q matrix.
        """
        return np.dot(f(self.x).T, self.w) # Q x 1

    def _to_string(self, pad):
        """Return a string representation of the object."""
        s = super(QuadratureRule, self)._to_string(pad) + '\n'
        s += pad + ' x: ' + str(self.x) + '\n'
        s += pad + ' w: ' + str(self.w) + '\n'
        return s