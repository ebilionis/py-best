"""Defines the concept of a real multi-input/output function.

It provides objects and methods that allow one to combine functions together
in order to construct new ones.

Author:
    Ilias Bilionis

Date:
    12/15/2012

"""


__all__ = ['Function', 'FunctionSum', 'FunctionMultiplication',
           'FunctionJoinedOutputs',
           'ConstantFunction', 'FunctionScreened',
           'GeneralizedLinearModel',
           'CovarianceFunction', 'CovarianceFunctionSum',
           'CovarianceFunctionProduct', 'CovarianceFunctionSE',
           'CovarianceFunctionBasis']



from ._function import *
from ._generalized_linear_model import *
from ._covariance_function import *