"""A routine useful for resampling.

Author:
    Ilias Bilionis

Date:
    2/5/2013

"""

import numpy as np


def multinomial_resample(p):
    """Sample the multinomial according to p and return a set of indices."""
    births = np.random.multinomial(p.shape[0], p)
    idx_list = []
    for i in xrange(p.shape[0]):
        idx_list += [i] * births[i]
    return idx_list
