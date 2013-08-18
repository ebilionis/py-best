"""A class representing a covariance function.

Author:
    Ilias Bilionis

Date:
    8/14/2013
"""


import numpy as np
import math
import itertools
from function import Function


class CovarianceFunction(object):

    """A class representing a covariance function."""

    # Number of inputs
    _num_input = None

    # A name
    _name = None

    # A wrapped function
    _k_wrapped = None

    # Parameters for the covariance function (1D numpy array)
    _num_hyp = None

    # Stored parameters (if any)
    _hyp = None

    @property
    def num_input(self):
        return self._num_input

    @property
    def name(self):
        return self._name

    @property
    def k_wrapped(self):
        return self._k_wrapped

    @property
    def is_wrapped(self):
        return not self.k_wrapped is None

    @property
    def hyp(self):
        return self._hyp

    @hyp.setter
    def hyp(self, hyp):
        hyp = np.atleast_1d(hyp)
        if not self._check_hyp(hyp):
            raise TypeError()
        self._hyp = hyp.copy()

    @property
    def is_hyp_set(self):
        return not self.hyp is None

    @property
    def num_hyp(self):
        return self._num_hyp

    def _check_hyp(self, hyp):
        """Check if the parameters are ok."""
        return len(hyp.shape) == 1 and hyp.shape[0] == self.num_hyp

    def __init__(self, num_input, num_hyp=0, hyp=None,
                 k_wrapped=None,
                 name='Covariance Function'):
        """Initialize the object.

        Arguments:
            num_input       ---     The number of inputs.

        Keyword Arguments
            num_hyp         ---     The number of hidden parameters.
                                    It is ignored if hyp is already set.
            hyp             ---     A vector for the hyper-parameters.
            k_wrapped       ---     A common function around which,
                                    the covariance function is wrapped.
            name            ---     A name for the covariance function.
        """
        assert isinstance(num_input, int)
        assert num_input >= 1
        self._num_input = num_input
        if not hyp is None:
            self._num_hyp = hyp.shape[0]
            self.hyp = hyp
        else:
            self._num_hyp = num_hyp
        assert isinstance(name, str)
        self._name = name
        if not k_wrapped is None:
            self._k_wrapped = k_wrapped

    def _gen_eval(self, x, y, func, num_out=1, hyp=None):
        # Check the hidden parameters
        if hyp is None:
            hyp = self.hyp
        if hyp is None:
            raise ValueError('You must specify the hyper-parameters.')
        hyp = np.atleast_1d(hyp)
        if not self._check_hyp(hyp):
            raise ValueError('Wrong number of parameters.')
        # If x and y are the same, then the resulting matrix will be
        # symmetric. Exploit this fact.
        is_sym = x is y
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        if self.num_input == 1:
            if not x.shape[1] == 1:
                x = x.T
            if not y.shape[1] == 1:
                y = y.T
        A = np.ndarray((x.shape[0], y.shape[0], num_out))
        if is_sym:
            # Symmetric version
            for i in range(x.shape[0]):
                for j in range(i + 1):
                    A[i, j, :] = func(x[i, :], x[j, :], hyp)
                    A[j, i, :] = A[i, j, :]
        else:
            # Non-symmetric version
            for i in range(x.shape[0]):
                for j in range(y.shape[0]):
                    A[i, j, :] = func(x[i, :], y[j, :], hyp)
        #if x.shape[0] == 1:
        #    if y.shape[0] == 1:
        #        return A[0, 0, :]
        #    return A[0, :, :]
        #if y.shape[0] == 1:
        #    return A[:, 0, :]
        return A

    def __call__(self, x, y, hyp=None):
        """Evaluate the covariance function at x and y.

        Arguments:
            x       ---     Input points.
            y       ---     Input points.

        Keyword Arguments:
            hyp     ---     The hidden parameter of the covariance
                            function. If None, then we look at the
                            parameters stored locally. If the local
                            hidden parameters are not set, then it is
                            assumed that num_hyp is zero.
        """
        A = self._gen_eval(x, y, self._eval, 1, hyp=hyp)
        return A.reshape(A.shape[:-1])

    def _eval(self, x, y, hyp):
        """Evaluate the covariance function at x and y given hyp."""
        if self.is_wrapped:
            return self.k_wrapped(x, y, hyp)
        else:
            raise NotImplementedError()

    def d(self, x, y, hyp=None):
        """Evaluate the Jacobian of the covariance function given hyp."""
        return self._gen_eval(x, y, self._d_eval, self.num_input, hyp=hyp)

    def _d_eval(self, x, y, hyp):
        """Evaluate the Jacobian of the covariance function given hyp.

        The Jacobian is with respect to y.
        """
        raise NotImplementedError()

    def d_hyp(self, x, y, hyp=None):
        return self._gen_eval(x, y, self._d_hyp_eval, self.num_hyp, hyp=hyp)

    def _d_hyp_eval(self, x, y, hyp):
        """Evaluate the derivative with respect to hyp."""
        raise NotImplementedError()

    def __str__(self):
        """Return a string representation of the object."""
        s = 'covariance Function: ' + self.name + '\n'
        s += 'num_input: ' + str(self.num_input) + '\n'
        s += 'num_hyp: ' + str(self.num_hyp)
        if self.is_hyp_set:
            s += '\nhyp:\n' + str(self.hyp)
        return s

    def __add__(self, cov):
        """Add two covariance functions."""
        return CovarianceFunctionSum((self, cov))

    def __mul__(self, cov):
        """Multiply two covariance functions."""
        return CovarianceFunctionProduct((self, cov))

    def to_basis(self, X, hyp=None, name='Covariance Function Basis'):
        """Construct a basis object from the cov.

        Arguments as in the constructor of CovarianceFunctionBasis.
        """
        return CovarianceFunctionBasis(self, X, hyp=hyp, name=name)


class _CovarianceFunctionContainer(CovarianceFunction):

    """A container for covariance functions."""

    # The covariance functions
    _cov = None

    # Store indices that let us identify the starting point
    # of the hyper-parameters for each cov in the global hyp array
    _start_hyp = None

    @property
    def cov(self):
        return self._cov

    @property
    def start_hyp(self):
        return self._start_hyp

    def __init__(self, cov, hyp=None,
                 name='covariance Function Container'):
        """Initialize the object.

        Arguments:
            cov     ---     A collection of RBFs.

        Keyword Argument
            hyp     ---     The hyper-parameters.
            name    ---     A name.
        """
        assert isinstance(cov, tuple) or isinstance(cov, list)
        assert len(cov) >= 1
        num_input = cov[0].num_input
        num_hyp = 0
        start_hyp = [0]
        for k in cov:
            assert isinstance(k, CovarianceFunction)
            assert num_input == k.num_input
            num_hyp += k.num_hyp
            start_hyp += [start_hyp[-1] + k.num_hyp]
        self._cov = cov
        self._start_hyp = start_hyp[:-1]
        super(_CovarianceFunctionContainer, self).__init__(num_input,
                                                            num_hyp=num_hyp,
                                                            hyp=hyp,
                                                            name=name)

    def __str__(self):
        """Get a string representation of the object."""
        s = super(_CovarianceFunctionContainer, self).__str__() + '\n'
        s += 'Contents:'
        for k in self.cov:
            s += '\n' + str(k)
        return s

    def _get_hyp_of(self, i, hyp):
        """Return the hyper-parameters pertaining to the i-th cov."""
        return hyp[self.start_hyp[i]: self.start_hyp[i] + self.cov[i].num_hyp]

    def _eval_all(self, x, y, func, hyp):
        """Evaluate func for all the covs in the container."""
        if hyp is None:
            hyp = self.hyp
        if hyp is None:
            raise ValueError('You must specify the hyper-parameters.')
        if not self._check_hyp(hyp):
            raise ValueError('Wrong number of parameters.')
        return [getattr(k, func)(x, y, hyp=self._get_hyp_of(i, hyp))
                for k, i in itertools.izip(self.cov, range(len(self.cov)))]


class CovarianceFunctionSum(_CovarianceFunctionContainer):

    """A container representing the sum of covariance functions."""

    def __init__(self, cov, hyp=None, name='Sum of covariance Functions'):
        """Initialize the object."""
        super(CovarianceFunctionSum, self).__init__(cov, hyp=hyp, name=name)

    def __call__(self, x, y, hyp=None):
        """Evaluate the cov."""
        return np.sum(self._eval_all(x, y, '__call__', hyp), axis=0)

    def d(self, x, y, hyp=None):
        """Evaluate the derivative of the cov."""
        return np.sum(self._eval_all(x, y, 'd', hyp), axis=0)

    def d_hyp(self, x, y, hyp=None):
        """Evaluate the derivative wrt the hyper-parameters."""
        tmp = self._eval_all(x, y, 'd_hyp', hyp)
        return np.concatenate(tmp, axis=-1)


class CovarianceFunctionProduct(_CovarianceFunctionContainer):

    """A container representing the product of covariance functions."""

    def __init__(self, cov, hyp=None, name='Product of covariance Functions'):
        """Initialize the object."""
        super(CovarianceFunctionProduct, self).__init__(cov, hyp=hyp, name=name)

    def __call__(self, x, y, hyp=None):
        """Evaluate the cov."""
        return np.prod(self._eval_all(x, y, '__call__', hyp), axis=0)

    def d(self, x, y, hyp=None):
        """Evaluate the derivative of the cov."""
        val = self._eval_all(x, y, '__call__', hyp)
        d_val = self._eval_all(x, y, 'd', hyp)
        res = np.zeros(d_val[0].shape)
        for i in range(len(val)):
            res += (np.prod(val[:i], axis=0) * d_val[i] *
                    np.prod(val[i + 1:], axis=0))
        return res

    def d_hyp(self, x, y, hyp=None):
        """Evaluate the derivative wrt the hyper-parameters."""
        tmp = self._eval_all(x, y, 'd_hyp', hyp)
        return np.concatenate(tmp, axis=-1)


class CovarianceFunctionSE(CovarianceFunction):

    """A Square Exponential covariance function."""

    def __init__(self, num_input, hyp=None, name='SE covariance Function'):
        """Initialize the object."""
        super(CovarianceFunctionSE, self).__init__(num_input, num_input,
                                                    hyp=hyp, name=name)

    def _eval(self, x, y, hyp):
        """Evaluate the function at x, y."""
        return math.exp(-0.5 * (((x - y) / hyp) ** 2).sum())

    def _d_eval(self, x, y, hyp):
        """Evaluate the derivative at x, y."""
        tmp = self._eval(x, y, hyp)
        return ((x - y) / hyp) * tmp / hyp

    def _d_hyp_eval(self, x, y, hyp):
        """Evaluate the derivative at x, y with respect to hyp."""
        tmp = self._eval(x, y, hyp)
        return ((x - y) / hyp) ** 2 * tmp / hyp


class CovarianceFunctionBasis(Function):

    """A basis built out of covariance functions."""

    # The covariance function object
    _cov = None

    # The fixed points of the basis
    _X = None

    # The hyper-parameters of the object.
    _hyp = None

    @property
    def cov(self):
        return self._cov

    @property
    def X(self):
        return self._X

    @property
    def hyp(self):
        return self._hyp

    def __init__(self, cov, X, hyp=None,
                 name='Covariance Function Basis'):
        """Initialize the object.

        Arguments:
            cov     ---     A CovarianceFunction object with set
                            hyper-parameters.
            X       ---     Input points for setting the first variable.

        Keyword Arguments
            hyp     ---     The hyper-parameters you wish to use. If None,
                            then we look at cov for them.
            name    ---     A name for the basis.
        """
        assert isinstance(cov, CovarianceFunction)
        assert cov.is_hyp_set or not hyp is None
        self._cov = cov
        if not hyp is None:
            hyp = np.array(hyp)
            self._hyp = hyp
        else:
            self._hyp = self.cov.hyp.copy()
        X = np.atleast_2d(X)
        assert isinstance(X, np.ndarray)
        assert len(X.shape) == 2
        if cov.num_input == 1 and not X.shape[1] == 1:
            X = X.T
        assert X.shape[1] == cov.num_input
        self._X = X
        super(CovarianceFunctionBasis, self).__init__(cov.num_input,
                                                       X.shape[0],
                                                       name=name)

    def _eval(self, x):
        """Evaluate the basis at x."""
        return self.cov(self.X, x, hyp=self.hyp)

    def _d_eval(self, x):
        """Evaluate the Jacobian of the basis at x."""
        return self.cov.d(self.X, x, hyp=self.hyp)

    def __str__(self):
        """Return a string representation of the object."""
        s = super(CovarianceFunctionBasis, self).__str__() + '\n'
        s += str(self.cov)
        return s