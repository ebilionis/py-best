"""A (truely) sparse Gaussian process.

The idea is to use a compact support covariance function.

Author:
    Ilias Bilionis

Date:
    4/2/2013
"""

import numpy as np
import scipy.sparse as sp
import scipy.optimize as opt
import scipy.spatial as dist
import math


def get_sparse_distance_diagonal_core(V, X, compute_der=False, eps=1e-16):
    n, dim = X.shape
    if compute_der:
        dx = np.zeros((n * (n - 1) / 2, dim))
        for j in range(dim):
            dx[:, j] = dist.distance.pdist(X[:, j:(j + 1)], 'seuclidean', V=V[j:(j + 1)])
        d = np.sqrt(np.sum(dx ** 2, axis=1))
    else:
        d = dist.distance.pdist(X, 'seuclidean', V=V)
        dx = None
    I = d >= 1
    d[I] = 0
    D = dist.distance.squareform(d) + eps * np.eye(n)
    Ds = sp.csc_matrix(D)
    if dx is not None:
        A = []
        dx[I, :] = 0
        for j in range(dim):
            Aj = dist.distance.squareform(dx[:, j]) + eps * np.eye(n)
            A.append(sp.csc_matrix(Aj))
    else:
        A = None
    return Ds, A


def get_sparse_distance_non_diagonal_core(V, X1, X2):
    D = dist.distance.cdist(X1, X2, 'seuclidean', V=V)
    return D


def get_sparse_distance_core(V, X, compute_der=False):
    return get_sparse_distance_diagonal_core(V, X, compute_der=compute_der)


def get_sparse_cross_covariance(hyp, X, Xt):
    dim = X.shape[1]
    ell = np.exp(hyp[:dim])
    sig = np.exp(hyp[dim])
    V = ell ** 2
    D = get_sparse_distance_non_diagonal_core(V, X, Xt)
    C = (2. + np.cos(2. * math.pi * D)) / 3. * (1. - D) + 1. / (2. * math.pi) * np.sin(2. * math.pi * D)
    C[D >= 1] = 0.
    return C


def get_sparse_covariance(hyp, X, max_mem=100, compute_der=False):
    dim = X.shape[1]
    ell = np.exp(hyp[:dim])
    sig = np.exp(hyp[dim])
    g = np.exp(hyp[dim + 1])
    V = ell ** 2
    Ds, A = get_sparse_distance_core(V, X, compute_der=compute_der)
    r = Ds.data
    r2pi = (2. * math.pi) * r
    cosr2pi = np.cos(r2pi)
    c1data = 2. + cosr2pi
    c1data /= 3.
    c1data *= 1. - r
    sinr2pi = np.sin(r2pi)
    c2data = sinr2pi
    c2data /= (2. * math.pi)
    c1data += c2data
    if len(Ds.indices) == 0:
        Cs = sp.spdiags(np.ones(Ds.shape[0]), 0, Ds.shape[0], Ds.shape[1], format='csc')
    else:
        print 'changing matrix'
        Cs = sp.csc_matrix((c1data, Ds.indices, Ds.indptr))
        #Cs = Cs.tolil()
        #Cs.setdiag(np.ones(Cs.shape[0]))
        #Cs = Cs.tocsc()
        print 'done'
    # Compute derivatives
    if compute_der:
        dCsdsig = sp.csc_matrix(Cs)
        cdata = math.pi * (r - 1.) * sinr2pi + cosr2pi - 1.
        cdata /= r
        cdata *= (-2. / 3. * sig)
        for j in range(dim):
            A[j] /= ell[j]
            A[j].data *= c1data
        A = A + [dCsdsig]
    else:
        dCsdsig = None
        A = None
    Cs *= sig
    Cs = Cs + g * sp.eye(*Ds.shape, format='csc')
    return Cs, A


def grad_sparse_gp_like(hyp, args):
    X = args['X']
    Y = args['Y']
    num_tr = args['num_tr']
    n, dim = X.shape
    out_dim = Y.shape[1]
    ell = np.exp(hyp[:dim])
    sig = np.exp(hyp[dim])
    g = np.exp(hyp[dim + 1])
    Cs, A = get_sparse_covariance(hyp, X, max_mem=100, compute_der=True)
    lu = sp.linalg.splu(Cs)
    KiY = np.zeros(Y.shape)
    for j in range(out_dim):
        KiY[:, j] = lu.solve(Y[:, j])
    grad = np.zeros(dim + 2)
    for j in range(dim + 1):
        grad[j] = 0.
        for q in range(out_dim):
            tmp = A[j].dot(KiY[:, q])
            grad[j] += np.dot(KiY[:, q].T, tmp)
        grad[j] /= out_dim
        trace = 0.
        for i in range(num_tr):
            d = np.random.randn(n)
            z1 = lu.solve(d)
            z2 = A[j].dot(d)
            trace += np.dot(z1, z2)
        trace /= num_tr
        grad[j] -= trace
        grad[j] *= 0.5
    grad[-1] = 0.
    for q in range(out_dim):
        grad[-1] += np.dot(KiY[:, q], KiY[:, q])
    grad[-1] /= out_dim
    grad[-1] -= Cs.diagonal().sum()
    grad[-1] *= 0.5
    grad *= np.exp(hyp) / n
    print '***'
    print np.exp(hyp)
    print grad
    print np.linalg.norm(grad)
    return grad


def get_sparse_gp_prediction(Xt, X, Y, hyp, K_lu, compute_covariance=False):
    n = X.shape[0]
    nt = Xt.shape[0]
    out_dim = Y.shape[1]
    Kt = get_sparse_cross_covariance(hyp, X, Xt)
    M = np.zeros((nt, out_dim))
    for i in range(Y.shape[1]):
        tmp = K_lu.solve(Y[:, i])
        M[:, i] = Kt.transpose().dot(tmp)
    if compute_covariance:
        Ktt = get_sparse_covariance(hyp, Xt)[0]
        tmp = np.zeros((n, nt))
        for i in range(nt):
            z = K_lu.solve(Kt.getcol(i).toarray().flatten())
            tmp[:, i] = K_lu.solve(Kt.getcol(i).toarray().flatten())
        tmp = Kt.T.dot(tmp)
        C = Ktt - tmp
        return M, C
    else:
        return M