"""Construct an 1D upscaling map.

Author:
    Ilias Bilionis

Date:
    12/15/2012

"""

import numpy as np


def create1DUpscalingMap(x, X):
    """Construct an 1D upscaling map.
    
    This is essentially an average map represented as a matrix.
    
    Arguments:
        x   ---     The fine scale grid.
        X   ---     The coarse scale grid.
    
    Precondition:
        Both x and X are sorted, x[0] == X[0] and x[-1] == X[-1].
    
    Caution:
        The preconditions are not checked!
    
    Return:
        The upsacling map in the form of a matrix.

    """
    if not isinstance(x, np.ndarray):
        raise TypeError('x must be a numpy array.')
    if not isinstance(X, np.ndarray):
        raise TypeError('X must be a numpy array.')
    x_diff = np.diff(x)
    X_diff = np.diff(X)
    n = x.shape[0] - 1
    N = X.shape[0] - 1
    M = np.zeros((N, n), order='F')
    i = 0
    k = 0
    while i < N:
        l = np.nonzero(x < X[i + 1])[0].max()
        M[i, k] = x[k + 1] - X[i]
        M[i, (k + 1):l] = x_diff[(k + 1):l]
        M[i, l] = X[i + 1] - x[l]
        M[i, k:(l + 1)] /= X_diff[i]
        k = l + 1
        i += 1
    return M