"""Latin Hypercube Sampling.

Author:
    Ilias Bilionis

Date:
    12/2/2012

"""

import numpy as np
import uq.core


def lhs_seed():
    """Return a random seed that can be used in lhs."""
    return uq.core.get_seed()

def lhs(n, k, seed=lhs_seed()):
    """Fill an n x k array with latin hyper-cube samples and return it.

    Arguments:
        n       ---     Number of samples.
        k       ---     Number of dimensions.

    Keyword Arguments:
        seed    ---     The random seed.

    """
    X = np.ndarray((k, n), order='F')
    uq.core.lhs(X, seed)
    return X.T
