"""A SPGP factory.

Author:
    Ilias Bilionis

Date:
    3/14/2013
"""


__all__ = ['create_SPGPSurrogate', 'create_SPGPBayesianSurrogate']


import numpy as np
from . import SPGPSurrogate
from . import SPGPBayesianSurrogate


def create_SPGPSurrogate(X, Y, particle, y_mean=None):
    """Create an SPGP surrogate from a particle."""
    if y_mean is None:
        y_mean = np.zeros((1, Y.shape[1]))
    xb = particle['xb']
    hyp = np.hstack([particle['log_b'], particle['log_c'],
                     particle['log_sig']])
    return SPGPSurrogate(X, Y, xb, hyp, y_mean=y_mean)

def create_SPGPBayesianSurrogate(X, Y, particles, weights, y_mean=None):
    """Create an SPGP Bayesian surrogate."""
    if y_mean is None:
        y_mean = np.zeros((1, Y.shape[1]))
    surrogates = [create_SPGPSurrogate(X, Y, p, y_mean=y_mean) for p in particles]
    return SPGPBayesianSurrogate(surrogates, weights)
