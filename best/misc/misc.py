"""Routines that do not fit anywhere else.

Author:
    Ilias Bilionis

Date:
    8/11/2013
"""


import math
import numpy as np


def logsumexp(a):
    A = a.max()
    r = math.log(np.exp(a-A).sum())
    return A + r