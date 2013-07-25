"""Define a polynomial mean model of a given degree.

Author:
    Ilias Bilionis

Date:
    12/2/2012

"""

from uq.gp import MeanModel


class PolynomialMeanModel(MeanModel):
    """Defines a polynomial mean model."""

    # The degree of the model
    _d = None

    @property
    def d(self):
        """Get the degree of the model."""
        return self._d

    def __init__(self, k, d=0):
        """Initialize the object."""
        self._d = d
        super(PolynomialMeanModel, self).__init__(k, d+1)

    def __call__(self, X, H=None):
        """Calculate the design matrix."""
        n = X.shape[0]
        if not X.shape[1] == self.k:
            raise ValueError('The dimensions of X are not right.')
        return_H = False
        if H is None:
            return_H = True
            H = np.ndarray((n, self.m), order='F')
        count = 0
        d = 0
        while d <= seld.d:
            H[:, count] = X
