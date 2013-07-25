"""A factory that creates Discrete-Karhunen Loeve Expansions.

Author:
    Ilias Bilionis

Date:
    12/13/2012

"""

from uq.random import DiscreteKarhunenLoeveExpansion
import itertools
import numpy as np
import scipy.linalg


def create_DiscreteKarhunenLoeveExpansion(A, energy=0.95, k_max=None):
    """Create a Discrete Karhunen-Loeve Expansion.

    Arguments:
        A       ---     The covariance matrix.
    
    Keyword Arguments:
        energy  ---     The energy of the field you wish to retain.
        k_max   ---     The maximum number of eigenvalues to be computed.

    """
    if not isinstance(A, tuple):
        A = (A, )
    if k_max is None:
        k_was_not_given = True
    else:
        k_was_not_given = False
    if not isinstance(k_max, tuple):
        k_max = (k_max, ) * len(A)
    W = ()
    V = ()
    n = 1
    for a, k in itertools.izip_longest(A, k_max):
        # Compute the eigenvalues
        #w, v = np.linalg.eig(A)
        if k is None:
            lo = 0
            hi = a.shape[0] - 1
        else:
            lo = a.shape[0] - 1 - k
            hi = a.shape[0] - 1
        if scipy.sparse.issparse(a):
            w, v = scipy.sparse.linalg.eigsh(a, k)
        else:
            w, v = scipy.linalg.eigh(a, eigvals=(lo, hi))
        # Zero-out everything that is negative
        w[w < 0.] = 0.
        W += (w, )
        V += (v, )
        n *= a.shape[0]
    if len(A) == 1:
        w = W[0]
        v = V[0]
    elif len(A) == 2:
        w = np.kron(W[0], W[1])
        v = np.kron(V[0], V[1])
    # Order the eigenvalues
    idx = np.argsort(w)[::-1]
    w = w[idx]
    v = v[:, idx]
    if k_was_not_given:
        w_cum = np.cumsum(w) / w.sum()
        idx = np.nonzero(w_cum <= energy)[0]
        w = w[idx]
        v = v[:, idx]
    # Create and return the Karhunen-Loeve Expansion
    return DiscreteKarhunenLoeveExpansion(v, w)
