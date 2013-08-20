"""A separable mean model.

Author:
    Ilias Bilionis

Date:
    12/2/2012

"""


__all__ = ['SeparableMeanModel']


import numpy as np
import itertools as iter
from . import MeanModel


class SeparableMeanModel(MeanModel):
    """A separable mean model."""

    # The components
    _mean = None

    # The dimensions of each component
    _k_of = None

    # The number of basis functions per component
    _m_of = None

    @property
    def mean(self):
        """Get the component list."""
        return self._mean

    @property
    def k_of(self):
        """Get the dimensions of all components."""
        return self._k_of

    @property
    def m_of(self):
        """Get the number of basis functions of all components."""
        return self._m_of

    @property
    def s(self):
        """Get the number of components."""
        return len(self._mean)

    def __init__(self, mean):
        """Initialize the mean."""
        self._mean = mean
        k = 1
        m = 1
        self._k_of = ()
        self._m_of = ()
        for mean in self._mean:
            if not isinstance(mean, MeanModel):
                raise TypeError('mean must be a list of MeanModels.')
            k *= mean.k
            self._k_of += (mean.k, )
            m *= mean.m
            self._m_of += (mean.m, )

    def __call__(self, X, H=None):
        """Evaluate the object."""
        n_of = [x.shape[0] for x in X]
        n = np.prod(n_of)
        return_H = False
        if H is None:
            H = []
            for nn, m in iter.izip(n_of, self.m_of):
                H.append(np.ndarray((nn, m), order='F'))
            return_H = True
        for x, h, mean in iter.izip(X, H, self.mean):
            mean(x, H=h)
        if return_H:
            return H
