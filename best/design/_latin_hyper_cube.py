"""Latin Hypercube Sampling.

Author:
    Ilias Bilionis

Date:
    12/2/2012

"""


__all__ = ['latin_center', 'latin_edge', 'lhs', 'lhs_seed']


from ..core import design
import numpy as np
from ..core import lhs as _lhs
from ..core import get_seed


def _check_args(num_points, num_dim, seed):
    """Check if the arguments to the latin_*() functions are ok."""
    if seed is None:
        seed = design.get_seed()
    seed = int(seed)
    num_points = int(num_points)
    num_dim = int(num_dim)
    assert seed > 0
    assert num_points >= 1
    assert num_dim >= 1
    return num_points, num_dim, seed


def latin_center(num_points, num_dim, seed=None):
    num_points, num_dim, seed = _check_args(num_points, num_dim, seed)
    return design.latin_center(num_dim, num_points, seed).T


def latin_edge(num_points, num_dim, seed=None):
    num_points, num_dim, seed = _check_args(num_points, num_dim, seed)
    return design.latin_edge(num_dim, num_points, seed).T


def lhs_seed():
    """Return a random seed that can be used in lhs."""
    return get_seed()


def lhs(n, k, seed=lhs_seed()):
    """Fill an n x k array with latin hyper-cube samples and return it.

    Arguments:
        n       ---     Number of samples.
        k       ---     Number of dimensions.

    Keyword Arguments:
        seed    ---     The random seed.

    """
    X = np.ndarray((k, n), order='F', dtype='float64')
    _lhs(X, seed)
    return X.T