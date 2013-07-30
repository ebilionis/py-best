"""

.. module: best.linalg.kron_prod
   :synopsis: Matrix multiplication with Kronecker products.

.. moduleauthor: Ilias Bilionis <ebilionis@gmail.com>

"""


import numpy as np


def kron_prod(A, x):
    """Matrix multiplication with Kronecker products.

    The routine computes the product:
        .. math::
            \mathbf{y} = (\otimes_{i=1}^s \mathbf{A}_i) \mathbf{x},
            
    where :math:`\mathbf{A}_i` are suitable matrices.
    The characteristic of the routine is that it does not form the
    Kronecker product explicitly. Also, :math:`\mathbf{x}` can be a
    matrix of appropriate dimensions.

    :param A: A collection of matrices whose Kronecker product is assumed.
    :param x: A vector or a matrix.
    :returns: A numpy array containing the result of the computation.

    Here is an example:

    >>> A1 = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
    >>> A2 = A1
    >>> A = (A1, A2)
    >>> x = np.ones(A1.shape[1] * A2.shape[1])
    >>> y = best.linalg.kron_prod(A, x)
    """
    m = A[-1].shape[0]
    n = A[-1].shape[1]
    if len(x.shape) == 1:
        x = x.reshape((x.shape[0], 1), order='F')
    q = x.shape[1]
    X = x.reshape((n, x.shape[0] / n * q), order='F')
    Z = np.dot(A[-1], X) # m x (x.shape[0] / n * q)
    if len(A) == 1:
        return Z
    else:
        s = np.vsplit(Z.T, q)
        st = np.hstack(s)
        r = kron_prod(A[:-1], st)
        s = np.vsplit(r.T, q)
        j = np.hstack(s)
        y = j.reshape((np.prod(j.shape) / q, q), order='F')
        return y
