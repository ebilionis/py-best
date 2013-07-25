"""Define some function used in the description of a Stiefeld manifold.

Author:
    Ilias Bilionis
    
Date:
    12/29/2012
    
"""

import numpy as np
import scipy
import math

def stiefel_log(X0, X):
    """The logarithm map on a compact Stiefel manifold.
    
    Project X to the tangent space of X0.
    
    Return:
        The projection Gamma.
        
    """
    n = X0.shape[0]
    k = X0.shape[1]
    I = np.eye(n)
    Gamma = np.zeros((n, k))
    for j in range(k):
        X0j = X0[:, j].reshape((n, 1), order='F')
        Xj = X[:, j].reshape((n, 1), order='F')
        X0TX = np.dot(X0j.T, Xj)
        Q, R = np.linalg.qr(X0TX)
        B = np.dot(I - np.dot(X0j, X0j.T), Xj)
        # Solve the system A * R = B
        A = scipy.linalg.solve_triangular(R, B.T, lower=False, trans='T').T
        C = np.dot(A, Q.T)
        U, Sigma, VT = np.linalg.svd(C, full_matrices=False)
        Gamma[:, j:j+1] = np.dot(np.dot(U, np.diag(np.arctan(Sigma))), VT)
    return Gamma
    #n = X0.shape[0]
    #k = X0.shape[1]
    #I = np.eye(X.shape[0])
    #Gamma = np.zeros((n, k))
    # Append X with zeros if necessary
    #if X.shape[1] < k:
    #    X = np.hstack((X, X0[:, X.shape[1]:]))
    # Assuming that all eigenvalues are distinct,
    # loop over columns
    #for j in range(k):
    #    X0jTX0j = np.dot(X0[:, j].T, X0[:, j])
    #    X0jTXj = np.dot(X0[:, j].T, X[:, j])
    #    A = np.dot(I - np.dot(X0[:, j], X0[:, j].T)/X0jTX0j,
    #               X[:, j] / X0jTXj * math.sqrt(X0jTX0j)).reshape((n, 1))
    #    U, Sigma, VT = np.linalg.svd(A, full_matrices=False)
    #    Gamma[:, j:j+1] = np.dot(U,
    #                             np.dot(np.diag(np.arctan(Sigma)),
    #                                    VT)
    #                            )
    #return Gamma
    
    # Rafael
    # Loop over columns
    #for j in range(k):
    #    A = np.dot(X0[:, j].T, X[:, j])
    #    B = np.dot(I - np.dot(X0[:, j], X0[:, j].T), X[:, j])
    #    Y = np.dot(B, A.T).reshape((n, 1))
    #    U, Sigma, VT = np.linalg.svd(Y, full_matrices=False)
    #    Gamma[:, j:j+1] = np.dot(U, np.dot(
    #                    np.diag(np.arctan(Sigma)),
    #                    VT))
    #return Gamma


def stiefel_exp(X0, Gamma):
    """The exponential map.
    
    Project Gamma to the manifold.
    
    Return:
        The projection X.
    
    """
    n = X0.shape[0]
    k = X0.shape[1]
    X = np.zeros((n, k))
    for j in range(k):
        X0j = X0[:, j].reshape((n, 1), order='F')
        Gammaj = Gamma[:, j].reshape((n, 1), order='F')
        U, Sigma, VT = np.linalg.svd(Gammaj, full_matrices=False)
        X[:, j:j+1] = (np.dot(X0j, np.dot(VT.T, np.diag(np.cos(Sigma))))
             + np.dot(U, np.diag(np.sin(Sigma))))
    return X
    #X = np.zeros((n, k))
    ## Append Gamma with zeros if necessary
    #if Gamma.shape[1] < k:
    #    Gamma = np.hstack((Gamma, np.zeros((n, k - Gamma.shape[1]))))
    # Loop over the eigenvalues
    #for j in range(k):
    #    U, Sigma, VT = np.linalg.svd(Gamma[:, j].reshape((n, 1)),
    #                                 full_matrices=False)
    #    X0jTX0j = np.dot(X0[:, j].T, X0[:, j])
    #    X[:, j] = (X0[:, j] / math.sqrt(X0jTX0j)
    #                     * VT[0, 0] * math.cos(Sigma[0])
    #               + U[:, 0] * math.sin(Sigma[0]))
    #return X
    # Rafael
    #X = (np.dot(X0, np.dot(
    #                VT.T,
    #                np.diag(np.cos(Sigma))
    #                )
    #            )
    #     + np.dot(
    #        U,
    #        np.diag(np.sin(Sigma))
    #        )
    #     )
    #return X