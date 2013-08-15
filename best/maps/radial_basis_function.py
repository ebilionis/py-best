"""A class representing a radial basis function.

Author:
    Ilias Bilionis

Date:
    8/14/2013
"""


import numpy as np
import math
import itertools
from function import Function


class RadialBasisFunction(object):

    """A class representing a radial basis function."""

    # Number of inputs
    _num_input = None

    # A name
    _name = None

    # A wrapped function
    _k_wrapped = None

    # Parameters for the radial basis function (1D numpy array)
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
                 name='Radial Basis Function'):
        """Initialize the object.

        Arguments:
            num_input       ---     The number of inputs.

        Keyword Arguments
            num_hyp         ---     The number of hidden parameters.
                                    It is ignored if hyp is already set.
            hyp             ---     A vector for the hyper-parameters.
            name            ---     A name for the radial basis function.
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
        if x.shape[0] == 1:
            if y.shape[0] == 1:
                return A[0, 0, :]
            return A[0, :, :]
        if y.shape[0] == 1:
            return A[:, 0, :]
        return A

    def __call__(self, x, y, hyp=None):
        """Evaluate the radial basis function at x and y.

        Arguments:
            x       ---     Input points.
            y       ---     Input points.

        Keyword Arguments:
            hyp     ---     The hidden parameter of the radial basis
                            function. If None, then we look at the
                            parameters stored locally. If the local
                            hidden parameters are not set, then it is
                            assumed that num_hyp is zero.
        """
        A = self._gen_eval(x, y, self._eval, 1, hyp=hyp)
        return A.reshape(A.shape[:-1])

    def _eval(self, x, y, hyp):
        """Evaluate the radial basis function at x and y given hyp."""
        if self.is_wrapped:
            return self.k_wrapped(x, y, hyp)
        else:
            raise NotImplementedError()

    def d(self, x, y, hyp=None):
        """Evaluate the Jacobian of the radial basis function given hyp."""
        return self._gen_eval(x, y, self._d_eval, self.num_input, hyp=hyp)

    def _d_eval(self, x, y, hyp):
        """Evaluate the Jacobian of the radial basis function given hyp.

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
        s = 'Radial Basis Function: ' + self.name + '\n'
        s += 'num_input: ' + str(self.num_input) + '\n'
        s += 'num_hyp: ' + str(self.num_hyp)
        if self.is_hyp_set:
            s += '\nhyp:\n' + str(self.hyp)
        return s

    def __add__(self, rbf):
        """Add two radial basis functions."""
        return RadialBasisFunctionSum((self, rbf))

    def __mul__(self, rbf):
        """Multiply two radial basis functions."""
        return RadialBasisFunctionProduct((self, rbf))


class _RadialBasisFunctionContainer(RadialBasisFunction):

    """A container for radial basis functions."""

    # The radial basis functions
    _rbf = None

    # Store indices that let us identify the starting point
    # of the hyper-parameters for each rbf in the global hyp array
    _start_hyp = None

    @property
    def rbf(self):
        return self._rbf

    @property
    def start_hyp(self):
        return self._start_hyp

    def __init__(self, rbf, hyp=None,
                 name='Radial Basis Function Container'):
        """Initialize the object.

        Arguments:
            rbf     ---     A collection of RBFs.

        Keyword Argument
            hyp     ---     The hyper-parameters.
            name    ---     A name.
        """
        assert isinstance(rbf, tuple) or isinstance(rbf, list)
        assert len(rbf) >= 1
        num_input = rbf[0].num_input
        num_hyp = 0
        start_hyp = [0]
        for k in rbf:
            assert isinstance(k, RadialBasisFunction)
            assert num_input == k.num_input
            num_hyp += k.num_hyp
            start_hyp += [start_hyp[-1] + k.num_hyp]
        self._rbf = rbf
        self._start_hyp = start_hyp[:-1]
        super(_RadialBasisFunctionContainer, self).__init__(num_input,
                                                            num_hyp=num_hyp,
                                                            hyp=hyp,
                                                            name=name)

    def __str__(self):
        """Get a string representation of the object."""
        s = super(_RadialBasisFunctionContainer, self).__str__() + '\n'
        s += 'Contents:'
        for k in self.rbf:
            s += '\n' + str(k)
        return s

    def _get_hyp_of(self, i, hyp):
        """Return the hyper-parameters pertaining to the i-th rbf."""
        return hyp[self.start_hyp[i]: self.start_hyp[i] + self.rbf[i].num_hyp]

    def _eval_all(self, x, y, func, hyp):
        """Evaluate func for all the rbfs in the container."""
        if hyp is None:
            hyp = self.hyp
        if hyp is None:
            raise ValueError('You must specify the hyper-parameters.')
        if not self._check_hyp(hyp):
            raise ValueError('Wrong number of parameters.')
        return [getattr(k, func)(x, y, hyp=self._get_hyp_of(i, hyp))
                for k, i in itertools.izip(self.rbf, range(len(self.rbf)))]


class RadialBasisFunctionSum(_RadialBasisFunctionContainer):

    """A container representing the sum of radial basis functions."""

    def __init__(self, rbf, hyp=None, name='Sum of Radial Basis Functions'):
        """Initialize the object."""
        super(RadialBasisFunctionSum, self).__init__(rbf, hyp=hyp, name=name)

    def __call__(self, x, y, hyp=None):
        """Evaluate the rbf."""
        return np.sum(self._eval_all(x, y, '__call__', hyp), axis=0)

    def d(self, x, y, hyp=None):
        """Evaluate the derivative of the rbf."""
        return np.sum(self._eval_all(x, y, 'd', hyp), axis=0)

    def d_hyp(self, x, y, hyp=None):
        """Evaluate the derivative wrt the hyper-parameters."""
        tmp = self._eval_all(x, y, 'd_hyp', hyp)
        return np.concatenate(tmp, axis=-1)


class RadialBasisFunctionProduct(_RadialBasisFunctionContainer):

    """A container representing the product of radial basis functions."""

    def __init__(self, rbf, hyp=None, name='Product of Radial Basis Functions'):
        """Initialize the object."""
        super(RadialBasisFunctionProduct, self).__init__(rbf, hyp=hyp, name=name)

    def __call__(self, x, y, hyp=None):
        """Evaluate the rbf."""
        return np.prod(self._eval_all(x, y, '__call__', hyp), axis=0)

    def d(self, x, y, hyp=None):
        """Evaluate the derivative of the rbf."""
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


class RadialBasisFunctionSE(RadialBasisFunction):

    """A Square Exponential Radial Basis function."""

    def __init__(self, num_input, hyp=None, name='SE Radial Basis Function'):
        """Initialize the object."""
        super(RadialBasisFunctionSE, self).__init__(num_input, num_input,
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

    def to_basis(self, X, hyp=None):
        """Turn the radial basis function into a basis."""
        if hyp is None:
            hyp = self.hyp
        return RadialBasisFunctionBasis(self, X)


class RadialBasisFunctionBasis(Function):

    """A basis built out of radial basis functions."""

    # The radial basis function object
    _rbf = None

    # The fixed points of the basis
    _X = None

    @property
    def rbf(self):
        return self._rbf

    @property
    def X(self):
        return self._X

    def __init__(self, rbf, X, name='Radial Basis Function Basis'):
        """Initialize the object.

        Arguments:
            rbf     ---     A RadialBasisFunction object with set
                            hyper-parameters.
            X       ---     Input points for setting the first variable.

        Keyword Arguments
            name    ---     A name for the basis.
        """
        assert isinstance(rbf, RadialBasisFunction)
        assert rbf.is_hyp_set
        self._rbf = rbf
        X = np.atleast_2d(X)
        assert isinstance(X, np.ndarray)
        assert len(X.shape) == 2
        if rbf.num_input == 1 and not X.shape[1] == 1:
            X = X.T
        assert X.shape[1] == rbf.num_input
        self._X = X
        super(RadialBasisFunctionBasis, self).__init__(rbf.num_input,
                                                       X.shape[0],
                                                       name=name)

    def _eval(self, x):
        """Evaluate the basis at x."""
        return self.rbf(self.X, x)

    def _d_eval(self, x):
        """Evaluate the Jacobian of the basis at x."""
        return self.rbf.d(self.X, x)

    def __str__(self):
        """Return a string representation of the object."""
        s = super(RadialBasisFunctionBasis, self).__str__() + '\n'
        s += str(self.rbf)
        return s