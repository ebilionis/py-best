"""
The GP model for KO.
"""


import pymc
import sys
sys.path.insert(0, '../..')
import best
import numpy as np
import scipy.linalg


def regularize(x):
    x = np.array(x)
    if x.ndim == 1:
        return x.reshape((x.shape[0], 1))
    return x


def regularize_tuple(X):
    if not (isinstance(X, tuple) or isinstance(X, list)):
        X = (X, )
    new_X = ()
    for x in X:
        new_X += (regularize(x), )
    return new_X


def make_model(X, Y):
    """
    Make the GP model with inputs ``X`` and observations ``Y``.
    """
    # Regularize the data
    X = regularize_tuple(X)
    Y = regularize(Y)

    # The total number of components
    s = len(X)

    # Get the number of samples per component
    n_of = [x.shape[0] for x in X]
    # The total number of samples
    n = np.prod(n_of)
    # Sanity check
    assert Y.shape[0] == n

    # The total number of outputs
    q = Y.shape[1]

    # Get the number of dimensions per component
    k_of = [x.shape[1] for x in X]
    # The total number of dimensions
    k = np.prod(k_of)

    # Start a covariance function for each component
    cov = [best.gp.SECovarianceFunction(k_of[i]) for i in range(len(k_of))]

    # For now on it only works with two components
    assert s == 2

    # Assign the hyper-parameters
    # The nuggets
    g1 = pymc.Exponential('g1', beta=1./1e-6, value=1e-6)
    g2 = pymc.Exponential('g2', beta=1./1e-6, value=1e-6)
    # The length scales
    ell1 = pymc.Exponential('ell1', beta=1./0.1, value=np.ones(k_of[0]) * 0.1)
    ell2 = pymc.Exponential('ell2', beta=1./0.1, value=np.ones(k_of[1]) * 0.1)

    # Deterministics that evaluate the covariance matrices
    @pymc.deterministic(plot=False)
    def cov_matrix1(value=None, ell1=ell1, cov1=cov[0], x1=X[0]):
        return cov1(ell1, x1)
    @pymc.deterministic(plot=False)
    def cov_matrix2(value=None, ell2=ell2, cov2=cov[1], x2=X[1]):
        return cov2(ell2, x2)
    # The covariance matrix with nuggets
    @pymc.deterministic(plot=False)
    def cov_matrix1_p_g1(value=None, cov_matrix1=cov_matrix1, g1=g1):
        return cov_matrix1 + g1 * np.eye(cov_matrix1.shape[0])
    @pymc.deterministic(plot=False)
    def cov_matrix2_p_g2(value=None, cov_matrix2=cov_matrix2, g2=g2):
        return cov_matrix2 + g2 * np.eye(cov_matrix2.shape[0])
    # The Cholesky decomposition of the covariance matrices
    @pymc.deterministic(plot=False)
    def chol1(value=None, cov_matrix1_p_g1=cov_matrix1_p_g1):
        try:
            L = np.linalg.cholesky(cov_matrix1_p_g1)
        except:
            L = np.zeros(cov_matrix1_p_g1.shape)
        return L
    @pymc.deterministic(plot=False)
    def chol2(value=None, cov_matrix2_p_g2=cov_matrix2_p_g2):
        try:
            L = np.linalg.cholesky(cov_matrix2_p_g2)
        except:
            L = np.zeros(cov_matrix2_p_g2.shape)
        return L
    # Finally, the logarithm of the likelihood of the model
    @pymc.stochastic(observed=True)
    def obs(value=Y, L1=chol1, L2=chol2, gamma=1.):
        # If we had problems computing any of the Cholesky's return zero
        # probability
        if np.sum(L1) == 0. or np.sum(L2) == 0.:
            return -np.inf
        # The dimensions
        s = 2
        n1 = L1.shape[0]
        n2 = L2.shape[0]
        n = n1 * n2
        m1 = 1
        m2 = 1
        m = 1
        q = Y.shape[1]
        # Compute the logarithms of the determinants of the Cholesky's
        log_det_L1 = 2. * np.log(np.diag(L1)).sum()
        log_det_L2 = 2. * np.log(np.diag(L2)).sum()
        H1 = np.ones((n1, m))
        H2 = np.ones((n2, m))
        H1s = scipy.linalg.solve_triangular(L1, H1, lower=True)
        H2s = scipy.linalg.solve_triangular(L2, H2, lower=True)
        QL1iH1, RL1iH1 = scipy.linalg.qr(H1s, mode='full')
        QL2iH2, RL2iH2 = scipy.linalg.qr(H2s, mode='full')
        log_det_H1TL1iH1 = 2. * np.log(np.abs(np.diag(RL1iH1))).sum()
        log_det_H2TL2iH2 = 2. * np.log(np.abs(np.diag(RL2iH2))).sum()
        # Solve some linear systems
        Ys = best.linalg.kron_solve((L1, L2), Y)
        # Compute B
        tmpm_n_q = best.linalg.kron_prod((QL1iH1.T, QL2iH2.T), Ys)
        B = tmpm_n_q[:m, :q]
        R1 = RL1iH1[:m, :m]
        R2 = RL2iH2[:m, :m]
        B = best.linalg.kron_solve((R1, R1), B)
        # Compute Sigma
        YmHBs = Ys - best.linalg.kron_prod((H1s, H2s), B)
        Sigma = (np.dot(YmHBs.T, YmHBs)) / (n - m)
        # Compute the Cholesky decomposition of Sigma
        LSigma = np.linalg.cholesky(Sigma)
        # Compute the determinant of Sigma
        log_det_Sigma = 2. * np.log(np.diag(LSigma)).sum()
        # The likelihood
        p1 = -0.5 * (n / n1) * q * log_det_L1 - 0.5 * (n / n2) * q * log_det_L2
        p2 = (-0.5 * (m / m1) * q * log_det_H1TL1iH1
              -0.5 * (m / m2) * q * log_det_H2TL2iH2)
        p3 = -0.5 * (n - m) * log_det_Sigma
        return gamma * (p1 + p2 + p3)

    return locals()
