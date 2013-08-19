"""

Author:
    Ilias Bilionis

Date:
    7/25/2013
"""


__all__ = ['lancz']


import numpy as np


def lancz(x, w, n, p0=None, p1=None):
    """The Lanczos procedure for constructing the recurcive formula
    for orthogonal polynomials.

    Reimplemented from Fortran (ORTHPOL).
    """
    ncap = x.shape[0]
    assert w.shape[0] == ncap
    if p0 == None:
        p0 = np.zeros(ncap)
    if p1 == None:
        p1 = np.zeros(ncap)
    if n <= 0:
        return
    p0[:] = x[:]
    p1[:] = 0.
    p1[0] = w[0]
    for i in range(ncap - 1):
        pi = w[i + 1]
        gam = 1.
        sig = 0.
        t = 0.
        xlam = x[i + 1]
        for k in range(i + 1):
            rho = p1[k] + pi
            tmp = gam * rho
            tsig = sig
            if rho <= 0.:
                gam = 1.
                sig = 0.
            else:
                gam = p1[k] / rho
                sig = pi / rho
            tk = sig * (p0[k] - xlam) - gam * t
            p0[k] += t - tk
            t = tk
            if sig <= 0.:
                pi = tsig * p1[k]
            else:
                pi = (t ** 2) / sig
            tsig = sig
            p1[k] = tmp
    return p0[:n], p1[:n]


if __name__ == '__main__':
    x = np.linspace(0., 1, 100)
    w = np.ones(100)
    print x, w
    alpha, beta = lancz(x, w, 10)
    print alpha
    print beta