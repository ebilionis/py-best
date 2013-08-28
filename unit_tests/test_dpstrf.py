from _dpstrf import *
import numpy as np


if __name__ == '__main__':
    n = 3
    A = np.ndarray((n, n), order='F', dtype='float64')
    A[:] = np.eye(n)
    A[0, 0] = 1.5
    A[1, 0] = 0.01
    A[1, 1]= 100.
    A[n-1, n-1] = 3.
    print A
    pinv = np.zeros(n, order='F', dtype='int32')
    pstrf('L', A, pinv, 1e-10)
    P = np.zeros((n, n), order='F', dtype='int32')
    P[pinv, range(n)] = 1
    print P
    L = np.dot(P, A)
    print np.dot(L, L.T)
