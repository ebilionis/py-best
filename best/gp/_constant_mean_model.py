"""A constant mean model.

Author:
    Ilias Bilionis

Date:
    12/2/2012

"""


__all__ = ['ConstantMeanModel']


import numpy as np
from . import MeanModel


class ConstantMeanModel(MeanModel):
    """A constant mean model."""

    def __init__(self, k):
        """Initialize the object."""
        super(ConstantMeanModel, self).__init__(k, 1)

    def __call__(self, X, H=None):
        """Evaluate the model."""
        n = X.shape[0]
        return_H = False
        if H is None:
            return_H = True
            H = np.ndarray((n, self.m), order='F')
        H.fill(1.)
        if return_H:
            return H
