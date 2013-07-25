"""Sparse Pseudo-Input Gaussian Processes.

Author:
    Ilias Bilionis

Date:
    2/18/2013

"""

import numpy as np
import scipy
import math


def dist(x0, x1):
    """Compute the pairwise distance matrix from column vectors."""
    n0 = x0.shape[0]
    n1 = x1.shape[0]
    x0 = np.atleast_2d(x0).T
    x1 = np.atleast_2d(x1).T
    return np.tile(x0, (1, n1)) - np.tile(x1.T, (n0, 1))


def spgp_lik(w, y, x, n, d=1e-6, compute_der=False, compute_der_xb=True):
    N = x.shape[0]
    dim = x.shape[1]
    xb = np.reshape(w[:-dim-2], (n, dim), order='F')
    b = np.exp(np.atleast_2d(w[-dim-1-1:-2]).T)
    c = math.exp(w[-2])
    sig = math.exp(w[-1])

    xb = xb * np.tile(np.sqrt(b).T, (n, 1))
    x = x * np.tile(np.sqrt(b).T, (N, 1))

    Q = np.dot(xb, xb.T)
    Q = (np.tile(np.atleast_2d(np.diag(Q)).T, (1, n))
         + np.tile(np.atleast_2d(np.diag(Q)), (n, 1)) - 2. * Q)
    Q = c * np.exp(-0.5 * Q) + d * np.eye(n)

    K = (-2. * np.dot(xb, x.T)
         + np.tile(np.atleast_2d(np.sum(x ** 2, 1)), (n, 1))
         + np.tile(np.atleast_2d(np.sum(xb ** 2, 1)).T, (1, N)))
    K = c * np.exp(-0.5 * K)

    L = scipy.linalg.cholesky(Q, lower=True)
    V = scipy.linalg.solve_triangular(L, K, lower=True)
    ep = 1. + (c - np.atleast_2d(np.sum(V ** 2, 0)).T) / sig
    K /= np.tile(np.sqrt(ep).T, (n, 1))
    V /= np.tile(np.sqrt(ep).T, (n, 1))
    # Copy of the data
    y = y / np.sqrt(ep)
    Lm = scipy.linalg.cholesky(sig * np.eye(n) + np.dot(V, V.T), lower=True)
    invLmV = scipy.linalg.solve_triangular(Lm, V, lower=True)
    bet = np.dot(invLmV, y)

    # Likelihood
    fw = (np.sum(np.log(np.diag(Lm))) + (N - n) / 2. * math.log(sig)
          + (np.dot(y.T, y) - np.dot(bet.T, bet)) / 2. / sig
          + np.sum(np.log(ep)) / 2. + 0.5 * N * math.log(2. * math.pi))

    # OPTIONAL DERIVATIVES
    if not compute_der:
        return fw[0, 0]

    # Precomputations
    Lt = np.dot(L, Lm)
    B1 = scipy.linalg.solve_triangular(Lt, invLmV, trans='T', lower=True)
    b1 = scipy.linalg.solve_triangular(Lt, bet, trans='T', lower=True)
    invLV = scipy.linalg.solve_triangular(L, V, trans='T', lower=True)
    invL = scipy.linalg.solve_triangular(L, np.eye(L.shape[0]), lower=True)
    invQ = np.dot(invL.T, invL)
    invLt = scipy.linalg.solve_triangular(Lt, np.eye(Lt.shape[0]), lower=True)
    invA = np.dot(invLt.T, invLt)
    tmp = scipy.linalg.solve_triangular(Lm, bet, trans=True, lower=True)
    mu = np.dot(tmp.T, V).T
    sumVsq = np.atleast_2d(np.sum(V ** 2, 0)).T
    bigsum = (y * np.dot(bet.T, invLmV).T / sig
              - np.atleast_2d(np.sum(invLmV ** 2, 0)).T / 2.
              - (y ** 2 + mu ** 2) / 2. / sig
              + 0.5)
    TT = np.dot(invLV, invLV.T * np.tile(bigsum, (1, n)))

    # Pseudo inputs and length scales
    dfxb = np.zeros((n, dim))
    dfb = np.zeros(dim)
    if compute_der_xb:
        for i in range(dim):
            dnnQ = dist(xb[:, i], xb[:, i]) * Q
            dNnK = dist(-xb[:, i], -x[:, i]) * K

            epdot = -2. / sig * dNnK * invLV
            epPmod = -np.atleast_2d(np.sum(epdot, 0)).T

            dfxb[:, i] = (- b1 * (np.dot(dNnK, (y - mu) / sig)
                                 + np.dot(dnnQ, b1))
                          + np.atleast_2d(np.sum((invQ - invA * sig) * dnnQ, 1)).T
                          + np.dot(epdot, bigsum)
                          - 2. / sig * np.atleast_2d(np.sum(dnnQ * TT, 1)).T)[:, 0]

            dfb[i] = np.dot(((y - mu).T * np.dot(b1.T, dNnK)) / sig
                + (epPmod * bigsum).T, x[:, i])

            dNnK *= B1
            dfxb[:, i] += np.sum(dNnK, 1)
            dfb[i] -= np.dot(np.atleast_2d(np.sum(dNnK, 0)), x[:, i])
            dfxb[:, i] *= math.sqrt(b[i])

            dfb[i] /= math.sqrt(b[i])
            dfb[i] += np.dot(dfxb[:, i].T, xb[:, i]) / b[i]
            dfb[i] *= math.sqrt(b[i]) / 2.

    # size
    epc = ( c / ep - sumVsq - d * np.atleast_2d(np.sum(invLV ** 2, 0)).T) / sig

    dfc = ((n + d * np.trace(invQ - sig * invA)
           - sig * np.sum(invA * Q.T)) / 2.
           - np.dot(mu.T, y - mu)[0, 0] / sig
           + np.dot(np.dot(b1.T, Q - d * np.eye(n)), b1) / 2.
           + np.dot(epc.T, bigsum))[0, 0]

    # noise
    dfsig = np.sum(bigsum / ep)
    dfw = np.hstack([dfxb.reshape((n * dim, ), order='F'),
                     dfb, dfc, dfsig])

    return fw[0, 0], dfw


def kern(x1, x2, hyp):
    n1, dim = x1.shape
    n2 = x2.shape[0]
    b = np.exp(hyp[:-2])
    c = math.exp(hyp[-2])
    # Copy of x1 x2
    x1 = x1 * np.tile(np.atleast_2d(np.sqrt(b)), (n1, 1))
    x2 = x2 * np.tile(np.atleast_2d(np.sqrt(b)), (n2, 1))

    K = (-2 * np.dot(x1, x2.T)
         + np.tile(np.atleast_2d(np.sum(x2 ** 2, 1)), (n1, 1))
         + np.tile(np.atleast_2d(np.sum(x1 ** 2, 1)).T, (1, n2)))
    K = c * np.exp(-0.5 * K)
    return K

def kdiag(x, hyp):

    c = math.exp(hyp[-2])
    Kd = np.tile(c, (x.shape[0], 1))
    return Kd

def spgp_pred(y, x, xb, xt, hyp, d=1e-6):
    N, dim = x.shape
    n = xb.shape[0]
    Nt = xt.shape[0]
    sig = math.exp(hyp[-1]) # Noise variance

    # precomputations
    K = kern(xb, xb, hyp) + d * np.eye(n)
    L = scipy.linalg.cholesky(K, lower=True)
    K = kern(xb, x, hyp)
    V = scipy.linalg.solve_triangular(L, K, lower=True)
    ep = 1. + (kdiag(x, hyp) - np.atleast_2d(np.sum(V ** 2, 0)).T) / sig
    V /= np.tile(np.sqrt(ep).T, (n, 1))
    # Copy y
    y = y / np.sqrt(ep)
    Lm = scipy.linalg.cholesky(sig * np.eye(n) + np.dot(V, V.T), lower=True)
    bet = scipy.linalg.solve_triangular(Lm, np.dot(V, y), lower=True)

    # test predictions
    K = kern(xb, xt, hyp)
    lst = scipy.linalg.solve_triangular(L, K, lower=True)
    lmst = scipy.linalg.solve_triangular(Lm, lst, lower=True)
    mu = np.dot(bet.T, lmst).T

    s2 = (kdiag(xt, hyp)
          - np.atleast_2d(np.sum(lst ** 2, 0)).T
          + sig * np.atleast_2d(np.sum(lmst ** 2, 0)).T)
        
    return mu, s2