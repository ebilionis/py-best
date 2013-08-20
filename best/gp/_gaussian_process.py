"""A Gaussian Process Class

Author:
    Ilias Bilionis

Date:
    8/18/2013
"""


class GaussianProcess(object):

    # Input points
    _X = None

    # Output points
    _Y = None

    # Covariance function
    _cov = None

    # Mean function
    _mean = None

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y

    @proprty
    def Y(self):
        return self._Y

    @property
    def cov(self):
        return self._cov

    @property
    def mean(self):
        return self._mean