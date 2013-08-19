"""Define a Discrete Karhunen-Loeve Expansion.

Author:
    Ilias Bilionis

Date:
    12/13/2012

"""


__all__ = ['KarhunenLoeveExpansion']


import numpy as np
from scipy.stats import norm
import scipy.linalg
import itertools
from ..domain import AllSpace
from ._random_vector import RandomVector


class KarhunenLoeveExpansion(RandomVector):

    """Define a Discrete Karhunen-Loeve Expansion.

    This can also be used to represent a PCA expansion.

    It can also be thought of a as a random vector.

    """

    # The set of eigen vectors
    _PHI = None

    # The set of eigen-values
    _lam = None

    # The mean of the model
    _mean = None

    # The signal strength of the model
    _sigma = None

    @property
    def PHI(self):
        """Get the eigen vectors."""
        return self._PHI

    @property
    def lam(self):
        """Get the eigen values."""
        return self._lam

    @property
    def sigma(self):
        """Get the signal strength of the model."""
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        """Set the signal strength of the model."""
        if not isinstance(value, float):
            raise TypeError('sigma must be a float.')
        if value < 0.:
            raise ValueError('sigma must be positive.')
        self._sigma = value

    @property
    def num_input(self):
        """Get the number of inputs."""
        return self._PHI.shape[1]

    @property
    def num_output(self):
        """Get the number of outputs."""
        return self._PHI.shape[0]

    def __init__(self, PHI, lam, mean=None, sigma=None,
                 name='Karhunen Loeve Expansion'):
        """Initialize the object.

        Arguments:
            PHI     ---     Should be the eigenvectors.
            lam     ---     Should be the eigenvalues.
            mean    ---     The mean of the model.
            sigma   ---     The signal strength of the model. The default value
                            is one.

        Precondition:
            PHI.shape[1] == lam.shape[1]

        """
        if not PHI.shape[1] == lam.shape[0]:
            raise ValueError('PHI and lam dimensions do not match.')
        self._PHI = PHI
        self._lam = lam
        if mean is None:
            mean = np.zeros(PHI.shape[0])
        mean = np.array(mean)
        assert mean.ndim == 1
        if not mean.shape[0] == self.PHI.shape[0]:
            raise ValueError('The dimensions of mean and PHI do not agree.')
        self._mean = mean
        if sigma is None:
            sigma = 1.
        self.sigma = sigma
        support = AllSpace(PHI.shape[0])
        super(KarhunenLoeveExpansion, self).__init__(support,
                                                     num_input=PHI.shape[1],
                                                     name=name)

    def _eval(self, theta, hyp):
        """Evaluate the Discrete Karhunen-Loeve Expansion.

        Arguments:
            theta   ---     The KL weights. The dimension of theta must be
                            less than the dimension of _lambda. If it is
                            strictly, less then the remaining coefficients are
                            assumed to be zero.

        Return:
            The sample.

        """
        m = theta.shape[0]
        assert m <= self.num_input
        return self._mean + np.dot(self.PHI[:, :m],
                                   self.sigma * np.sqrt(self.lam[:m]) *
                                   theta)

    def project(self, Y):
        """Project Y to the space of KL weights.

        Arguments:
            Y   ---     A sample of the field.

        Return:
            theta   ---     The KLE weights corresponding to Y.

        """
        return (np.dot(self.PHI.T, Y - self.mean()) /
                (np.sqrt(self.lam) * self.sigma))

    def _rvs(self):
        """Return a sample of the random vector."""
        return self(norm.rvs(size=self.num_input))

    def _pdf(self, y):
        """Return the pdf at a particular point."""
        return np.prod(norm.pdf(self.project(y)))

    def mean(self):
        """Get the mean of the model."""
        return self._mean

    @staticmethod
    def create_from_data(Y, energy=0.95):
        """Create the KLE from data."""
        raise NotImplementedError()

    @staticmethod
    def create_from_covariance_matrix(A, mean=None, energy=0.95, k_max=None):
        """Create a Discrete Karhunen-Loeve Expansion.

        Arguments:
            A       ---     The covariance matrix.

        Keyword Arguments:
            mean    ---     The mean of the model.
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
        return KarhunenLoeveExpansion(v, w, mean=mean)