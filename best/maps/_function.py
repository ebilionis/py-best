"""A class representing an arbitrary function.

Author:
    Ilias Bilionis

Date:
    12/15/2012

"""


__all__ = ['Function', 'FunctionSum', 'FunctionProduct',
           'FunctionJoinedOutputs',
           'ConstantFunction', 'FunctionScreened']


import numpy as np
import warnings
import itertools
from .. import Object


class Function(Object):

    """A class representing an arbitrary multi-input/output function.

    Everything in Best that can be thought of as a function should be
    a child of this class.
    """

    # Number of input dimensions
    _num_input = None

    # Number of output dimensions
    _num_output = None

    # If this is a function wrapper
    _f_wrapped = None

    # Parameters for the function
    _hyp = None

    # Number of parameters
    _num_hyp = None

    @property
    def num_input(self):
        """Get the number of input dimensions."""
        return self._num_input

    @property
    def num_output(self):
        """Get the number of output dimensions."""
        return self._num_output

    @property
    def f_wrapped(self):
        """Get the wrapped function (if any)."""
        return self._f_wrapped

    @property
    def is_function_wrapper(self):
        """Is this simply a function wrapper."""
        return self.f_wrapped is not None

    @property
    def hyp(self):
        """Get the hyper-parameters."""
        return self._hyp

    @hyp.setter
    def hyp(self, val):
        """Set the hyper-parameters."""
        if not val is None:
            val = np.array(val)
            assert val.ndim == 1
            assert val.shape[0] == self.num_hyp
        self._hyp = val

    @property
    def num_hyp(self):
        """Get the number of hyper-parameters"""
        return self._num_hyp

    def _check_if_valid_dim(self, value, name):
        """Checks if value is a valid dimension."""
        if not isinstance(value, int):
            raise TypeError(name + ' must be an integer.')
        if value < 0:
            raise ValueError(name + ' must be non-negative.')

    def _fix_x(self, x):
        """Fix x."""
        x = np.atleast_2d(x)
        if self.num_input == 1 and not x.shape[1] == 1:
            x = x.T
        return x

    def __init__(self, num_input, num_output, num_hyp=0, name='function',
                 f_wrapped=None):
        """Initializes the object

        Arguments:
            num_input   ---     The number of input dimensions.
            num_output  ---     The number of output dimensions.
            num_hyp     ---     The number of hyper-parameters.

        Keyword Arguments:
            name        ---     Provide a name for the function.
            f_wrapped   ---     If specified, then this function is simply
                                a wapper of f_wrapped.
        """
        self._check_if_valid_dim(num_input, 'num_input')
        self._num_input = num_input
        self._check_if_valid_dim(num_output, 'num_output')
        self._num_output = num_output
        num_hyp = int(num_hyp)
        assert num_hyp >= 0
        self._num_hyp = 0
        if f_wrapped is not None:
            if not hasattr(f_wrapped, '__call__'):
                raise TypeError('f_wrapped must be a proper function.')
            self._f_wrapped = f_wrapped
            if not hyp is None:
                warnings.warn('The specified hyper-parameters will be '
                              + 'ignored!')
        super(Function, self).__init__(name=name)

    def _eval(self, x, hyp):
        """Evaluate the function assume x has the right dimensions."""
        if self.is_function_wrapper:
            return self.f_wrapped(x)
        else:
            raise NotImplementedError()

    def _d_eval(self, x, hyp):
        """Evaluate the derivative of the function at x.

        It should return a 2D numpy array of dimensions
        num_output x num_input. That is, it should return
        the Jacobian at x.
        """
        raise NotImplementedError()

    def _d_hyp_evel(self, x, hyp):
        """Evaluate the derivative of the function with respect to hyp.

        It should return a 3D numpy array of dimensions
        num_output x num_input x num_dim.
        """
        raise NotImplementedError()

    def _gen_eval(self, x, func, hyp):
        """Evaluate either the function or its derivative."""
        x = self._fix_x(x)
        num_dim = x.shape[1]
        assert num_dim == self.num_input
        num_pt = x.shape[0]
        res = np.concatenate([np.atleast_2d(func(x[i, :], hyp))
                                            for i in range(num_pt)],
                             axis=0)
        if num_pt == 1:
            return res[0, :]
        return res

    def __call__(self, x, hyp=None):
        """Call the object.

        If y is provided, then the Function should write the output
        on y.

        Arguments:
            x       ---     The input variables.

        """
        return self._gen_eval(x, self._eval, hyp)

    def d(self, x, hyp=None):
        """Evaluate the derivative of the function at x."""
        return self._gen_eval(x, self._d_eval, hyp)

    def _to_func(self, obj):
        """Take func and return a proper function if it is a float."""
        if isinstance(obj, float) or isinstance(obj, int):
            obj = np.ones(self.num_output) * obj
        if isinstance(obj, np.ndarray):
            obj = ConstantFunction(self.num_input, obj)
        if not isinstance(obj, Function) and hasattr(obj, '__call__'):
            obj = Function(self.num_input, self.num_output, f_wrapped=obj)
        return obj

    def __add__(self, func):
        """Add this function with another one."""
        func = self._to_func(func)
        if self is func:
            return self * 2
        functions = (self, )
        if isinstance(self, FunctionSum):
            functions = self.functions
        functions += (func, )
        return FunctionSum(functions)

    def __mul__(self, func):
        """Multiply two functions."""
        func = self._to_func(func)
        if self is func:
            return FunctionPower(self, 2.)
        functions = (self, )
        if isinstance(self, FunctionProduct):
            functions = self.functions
        functions += (func, )
        return FunctionProduct(functions)

    def compose(self, func):
        """Compose two functions."""
        f = FunctionComposition((self, func))
        return f

    def _to_string(self, pad):
        """Return a padded string representation."""
        s = super(Function, self)._to_string(pad) + '\n'
        s += pad + ' f:R^' + str(self.num_input) + ' --> '
        s += ' R^' + str(self.num_output) + '\n'
        s += pad + ' num_hyp: ' + str(self.num_hyp)
        if self.is_function_wrapper:
            s += '\n' + pad + ' (function wrapper)'
        return s

    def join(self, func):
        """Join the outputs of two functions."""
        func = self._to_func(func)
        functions = (self, )
        if isinstance(self, FunctionJoinedOutputs):
            functions = self.functions
        functions += (func, )
        return FunctionJoinedOutputs(functions)

    def screen(self, in_idx=None, out_idx=None, default_inputs=None,
               name='Screened Function'):
        """Construct a screened version of the function object.

        Arguments:
            As in the constructor of FunctionScreened.
        """
        return FunctionScreened(self, in_idx=in_idx, out_idx=out_idx,
                                default_inputs=default_inputs,
                                name=name)


class _FunctionContainer(Function):

    """A collection of functions."""

    # The functions (a tuple)
    _functions = None

    # Store indices that let us identify the starting point
    # of the hyper-parameters for each cov in the global hyp array
    _start_hyp = None

    @property
    def functions(self):
        """Get the functions involved in the summation."""
        return self._functions

    @property
    def start_hyp(self):
        return self._start_hyp

    def __init__(self, functions, name='Function Container'):
        """Initialize the object.

        Aruguments:
            functions   ---     A tuple of functions.

        """
        if not isinstance(functions, tuple):
            raise TypeError('functions must be a tuple')
        for f in functions:
            if not isinstance(f, Function):
                raise TypeError(
                        'All members of functions must be Functions')
        num_input = functions[0].num_input
        num_output = functions[0].num_output
        num_hyp = functions[0].num_hyp
        start_hyp = [0]
        for f in functions[1:]:
            if num_input != f.num_input or num_output != f.num_output:
                raise ValueError(
                    'All functions must have the same dimensions.')
            num_hyp += f.num_hyp
            start_hyp += [start_hyp[-1] + f.num_hyp]
        self._functions = functions
        self._start_hyp = start_hyp[:-1]
        super(_FunctionContainer, self).__init__(num_input, num_output,
                                                  num_hyp=num_hyp,
                                                  name=name)

    def _to_string(self, pad):
        """Return a string representation with padding."""
        s = super(_FunctionContainer, self)._to_string(pad)
        s += pad + ' Contents:'
        for f in self.functions:
            s += '\n' + f._to_string(pad + ' ')
        return s

    def _get_hyp_of(self, i, hyp):
        """Return the hyper-parameters pertaining to the i-th cov."""
        return hyp[self.start_hyp[i]: self.start_hyp[i] + self.cov[i].num_hyp]

    def _eval_all(self, x, func, hyp):
        """Evaluate func for all the functions in the container."""
        return [getattr(f, func)(x, self._get_hyp_of(i, hyp))
                for f, i in itertools.izip(self.functions,
                                           range(len(functions)))]


class FunctionSum(_FunctionContainer):

    """Define the sum of functions."""

    def __init__(self, functions, name='Function Sum'):
        """Initialize the object."""
        super(FunctionSum, self).__init__(functions, name=name)

    def __call__(self, x, hyp=None):
        """Evaluate the function."""
        return np.sum(self._eval_all(x, '__call__', hyp), axis=0)

    def d(self, x, hyp=None):
        """Evaluate the derivative of the function."""
        return np.sum(self._eval_all(x, 'd', hyp), axis=0)


class FunctionJoinedOutputs(_FunctionContainer):

    """Define a function that joins the outputs of two functions."""

    def __init__(self, functions, name='Function Joined Outputs'):
        """Initialize the object."""
        if not isinstance(functions, tuple):
            raise TypeError('functions must be a tuple')
        for f in functions:
            if not isinstance(f, Function):
                raise TypeError(
                        'All members of functions must be Functions')
        num_input = functions[0].num_input
        num_output = functions[0].num_output
        num_hyp = functions[0].num_hyp
        for f in functions[1:]:
            if num_input != f.num_input:
                raise ValueError(
                    'All functions must have the same dimensions.')
            num_output += f.num_output
            num_hyp += f.num_hyp
        self._functions = functions
        super(_FunctionContainer, self).__init__(num_input, num_output,
                                                  num_hyp=num_hyp,
                                                  name=name)

    def __call__(self, x, hyp=None):
        return np.concatenate(self._eval_all(x, '__call__', hyp), axis=-1)

    def d(self, x, hyp=None):
        return np.concatenate(self._eval_all(x, 'd', hyp), axis=-1)


class FunctionProduct(_FunctionContainer):

    """Define the multiplication of functions (element wise)"""

    def __init__(self, functions, name='Function Product'):
        """Initialize the object."""
        super(FunctionProduct, self).__init__(functions, name=name)

    def __call__(self, x, hyp=None):
        """Evaluate the function at x."""
        return np.prod(self._eval_all(x, '__call__', hyp), axis=0)

    def d(self, x, hyp=None):
        """Evaluate the derivative of the function."""
        val = self._eval_all(x, '__call__', hyp)
        d_val = self._eval_all(x, 'd')
        res = np.zeros(d_val[0].shape)
        for i in range(len(self.functions)):
            for k in range(self.num_input):
                res[:, :, k] += np.prod((val[:i], val[(i + 1):],
                                         d_val[i][:, :, k]), axis=0)
        return res


class ConstantFunction(Function):

    """Define a constant function."""

    # The constant
    _const = None

    @property
    def const(self):
        """Get the constant."""
        return self._const

    def __init__(self, num_input, const, name='Constant Function'):
        """Initialize the object.

        Arguments:
            const   ---     Must be a numpy array.

        """
        if isinstance(const, float):
            num_output = 1
            c_a = np.ndarray(1, order='F')
            c_a[0] = const
            const = c_a
        elif isinstance(const, np.ndarray):
            num_output = const.shape[0]
        else:
            raise TypeError(
                'The constant must be either a float or an array.')
        super(ConstantFunction, self).__init__(num_input, num_output, name=name)
        self._const = const

    def _eval(self, x, hyp):
        """Evaluate the function at x."""
        return self.const

    def _d_eval(self, x, hyp):
        """Evaluate the derivative of the function at x."""
        return np.zeros(self.const.shape)


class FunctionComposition(Function):

    """Define the composition of a collection of functions."""

    # The functions
    _functions = None

    @property
    def functions(self):
        """Get the functions."""
        return self._functions

    def __init__(self, functions, name='Function composition'):
        """Initialize the object.

        Arguments:
            functions   ---     Function to be composed.
        """
        num_input = functions[-1].num_input
        num_output = functions[0].num_output
        if not isinstance(functions, tuple):
            raise TypeError('The functions must be provided as a tuple.')
        for i in xrange(1, len(functions)):
            if not functions[i-1].num_input == functions[i].num_output:
                raise ValueError('Dimensions of ' + str(functions[i-1]) +
                                 ' and '
                                 + str(functions[i]) + ' do not agree.')
        self._functions = functions
        super(FunctionComposition, self).__init__(num_input, num_output,
                                                  name=name)

    def __call__(self, x, hyp=None):
        """Evaluate the function at x."""
        z = x
        for f in self.functions[-1::-1]:
            z = f(z)
        return z

    def d(self, x, hyp=None):
        """Evaluate the derivative at x."""
        dg = self.functions[-1].d(x)
        df = self.functions[-2].d(dg)
        dh = np.einsum('ijk, ikm->ijm', df, dg)

    def _to_string(self, pad):
        """Return a string representation of the object."""
        s = super(FunctionComposition, self)._to_string(pad)
        for f in self.functions:
            s += '\n' + f._to_string(pad + ' ')
        return s


class FunctionPower(Function):

    """Raise a function to a given power."""

    # The underlying function object.
    _function = None

    # The exponent
    _exponent = None

    @property
    def function(self):
        """Get the funciton object."""
        return self._function

    @property
    def exponent(self):
        """Get the exponent."""
        return self._exponent

    def __init__(self, f, exponent, name='Function Power'):
        """Initialize the object.

        Arguments:
            f       ---     The underlying function.
            exponent---     The exponent to which you want to raise it.
        """
        if not isinstance(f, Function):
            raise TypeError('f must be a function.')
        self._function = f
        if not isinstance(exponent, float) and not isinstance(exponent, int):
            raise TypeError('The exponent must be an scalar.')
        self._exponent = exponent
        super(FunctionPower, self).__init__(f.num_input, f.num_output,
                                            num_hyp=f.num_hyp, name=name)

    def __call__(self, x, hyp=None):
        """Evaluate the function at x."""
        return self.function(x, hyp) ** self.exponent()

    def d(self, x, hyp=None):
        """Evaluate the function at x."""
        raise NotImplementedError()
        return self.function.d(x, hyp) ** (self.exponent() - 1.)


class FunctionScreened(Function):

    """Create a function with screened inputs/outputs."""

    # The screened function
    _screened_func = None

    # Input indices
    _in_idx = None

    # Default inputs
    _default_inputs = None

    # Output indices
    _out_idx = None

    @property
    def screened_func(self):
        return self._screened_func

    @property
    def in_idx(self):
        return self._in_idx

    @property
    def default_inputs(self):
        return self._default_inputs

    @property
    def out_idx(self):
        return self._out_idx

    def __init__(self, screened_func, in_idx=None, default_inputs=None,
                 out_idx=None, name='Screened Function'):
        """Initialize the object.

        Create a function with screened inputs and/or outputs.

        Arguments:
            screenedf_func ---  The function to be screened.

        Keyword Arguments:
            in_idx      ---     The input indices that are not screened.
                                It must be a valid container.
                                If None, then no inputs are screened. If
                                a non-empty list is suplied, then the
                                argument default_inputs must be suplied.
            default_inputs ---  If in_idx is not None, then this can be
                                suplied. It is a default set of inputs
                                of the same size as the original input
                                of the screened_func. If it is not
                                given, then it is automatically set to
                                zero. These values will be used to fill
                                in the missing values of the screened
                                inputs.
            out_idx     ---     The output indices that are not screened.
                                If None, then no output is screened.
            name        ---     A name for the function.
        """
        if not isinstance(screened_func, Function):
            raise TypeError('The screened_func must be a Function.')
        self._screened_func = screened_func
        if in_idx is None:
            in_idx = np.arange(self.screened_func.num_input, dtype='i')
        else:
            in_idx = np.array(in_idx, dtype='i')
            assert in_idx.ndim == 1
            assert in_idx.shape[0] <= self.screened_func.num_input
        self._in_idx = in_idx
        if default_inputs is None:
            default_inputs = np.zeros(self.screened_func.num_input)
        else:
            default_inputs = np.array(default_inputs)
            assert default_inputs.ndim == 1
            assert default_inputs.shape[0] == self.screened_func.num_input
        self._default_inputs = default_inputs
        if out_idx is None:
            out_idx = np.arange(self.screened_func.num_output, dtype='i')
        else:
            out_idx = np.array(out_idx, dtype='i')
            assert out_idx.ndim == 1
            assert out_idx.shape[0] <= self.screened_func.num_output
        self._out_idx = out_idx
        num_input = in_idx.shape[0]
        num_output = out_idx.shape[0]
        super(FunctionScreened, self).__init__(num_input, num_output,
                                               name=name)

    def _get_eval(self, x, func):
        """Evaluate func at x."""
        x = self._fix_x(x)
        x_full = np.array(x, copy=True)
        x_full[:, self.in_idx] = x
        y_full = func(x_full)
        if x.shape[0] == 1:
            return y_full[0, self.out_idx]
        return y_full[:, self.out_idx]

    def __call__(self, x):
        """Evaluate the function at x."""
        return self._get_eval(x, self.screened_func.__call__)

    def d(self, x):
        """Evaluate the derivative of the function at x."""
        return self._get_eval(x, self.screened_func.d)

    def _to_string(self, pad):
        """Return a padded string representation."""
        s = super(FunctionScreened, self)._to_string(pad) + '\n'
        s += self.screened_func._to_string(pad + ' ') + '\n'
        s += pad + ' in_idx: ' + str(self.in_idx) + '\n'
        s += pad + ' default_inputs: ' + str(self.default_inputs) + '\n'
        s += pad + ' out_idx: ' + str(self.out_idx)
        return s