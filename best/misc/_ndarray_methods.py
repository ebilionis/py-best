"""Methods that are related to numpy arrays.

Author:
    Ilias Bilionis

Date:
    8/11/2013
"""


__all__ = ['logsumexp']


import math
import numpy as np


def logsumexp(a):
    print str(best)
    A = a.max()
    r = math.log(np.exp(a-A).sum())
    return A + r