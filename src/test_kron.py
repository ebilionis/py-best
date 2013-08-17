from uq.core import *
import numpy as np

A = 2. * np.eye(5)
b = np.ones((5, 2), order='F')
trsm('L', 'L', 'N', 'N', 1., A, b)
print b
