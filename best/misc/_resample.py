"""A routine useful for resampling.

Author:
    Ilias Bilionis

Date:
    2/5/2013

"""


__all__ = ['multinomial_resample']


import numpy as np


def multinomial_resample(p):
    """Sample the multinomial according to p.

    Arguments:
        p       ---     A numpy array of positive numbers that sum to one.

    Return:
        A set of indices sampled according to p.
    """
    p = np.array(p)
    assert p.ndim == 1
    assert (p >= 0.).all()
    births = np.random.multinomial(p.shape[0], p)
    idx_list = []
    for i in xrange(p.shape[0]):
        idx_list += [i] * births[i]
    return np.array(idx_list, dtype='i')