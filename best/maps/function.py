"""A class representing an arbitrary function.

Author:
    Ilias Bilionis

Date:
    12/15/2012

"""

import numpy as np


class Function(object):

    """A class representing an arbitrary multi-input/output function.

    Everything in Best that can be thought of as a function should be
    a child of this class.
    """

    # A name for this function
    _name = None

    # Number of input dimensions
    _num_input = None

    # Number of output dimensions
    _num_output = None

    # If this is a function wrapper
    _f_wrapped = None

    @property
    def num_input(self):
        """Get the number of input dimensions."""
        return self._num_input

    @property
    def num_output(self):
        """Get the number of output dimensions."""
        return self._num_output

    @property
    def name(self):
        """Get the name of the function."""
        return self._name

    @name.setter
    def name(self, value):
        """Set the name of the function."""
        if not isinstance(value, str):
            raise TypeError('The name of the function must be a string.')
        self._name = value

    @property
    def f_wrapped(self):
        """Get the wrapped function (if any)."""
        return self._f_wrapped

    @property
    def is_function_wrapper(self):
        """Is this simply a function wrapper."""
        return self.f_wrapped is not None

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

    def __init__(self, num_input, num_output, name='function',
                 f_wrapped=None):
        """Initializes the object

        Arguments:
            num_input   ---     The number of input dimensions.
            num_output  ---     The number of output dimensions.

        Keyword Arguments:
            name        ---     Provide a name for the function.
            f_wrapped   ---     If specified, then this function is simply
                                a wapper of f_wrapped.
        """
        self._check_if_valid_dim(num_input, 'num_input')
        self._num_input = num_input
        self._check_if_valid_dim(num_output, 'num_output')
        self._num_output = num_output
        self.name = name
        if f_wrapped is not None:
            if not hasattr(f_wrapped, '__call__'):
                raise TypeError('f_wrapped must be a proper function.')
            self._f_wrapped = f_wrapped

    def _gen_eval(self, x, func):
        """Evaluate either the function or its derivative."""
        x = self._fix_x(x)
        size = x.shape[:-1]
        num_dim = x.shape[-1]
        assert num_dim == self.num_input
        num_pt = np.prod(size)
        x = x.reshape((num_pt, num_dim))
        res = np.concatenate([func(x[i, :]) for i in range(num_pt)],
                             axis=0)
        if self.num_output == 1:
            if num_pt == 1:
                return res[0, 0]
            return res[:, 0]
        if num_pt == 1:
            return res[0, :]
        shape = size + (self.num_output, )
        return res.reshape(shape)

    def __call__(self, x):
        """Call the object.

        If y is provided, then the Function should write the output
        on y.

        Arguments:
            x       ---     The input variables.

        """
        return self._gen_eval(x, self._eval)

    def _eval(self, x):
        """Evaluate the function assume x has the right dimensions."""
        if self.is_function_wrapper:
            return self.f_wrapped(x)
        else:
            raise NotImplementedError()

    def d(self, x):
        """Evaluate the derivative of the function at x."""
        return self._gen_eval(x, self._d_eval)

    def _d_eval(self, x):
        """Evaluate the derivative of the function at x.

        It should return a 2D numpy array of dimensions
        num_output x num_input. That is, it should return
        the Jacobian at x.
        """
        raise NotImplementedError()

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
        if isinstance(self, FunctionMultiplication):
            functions = self.functions
        functions += (func, )
        return FunctionMultiplication(functions)

    def compose(self, func):
        """Compose two functions."""
        f = FunctionComposition((self, func))
        return f

    def _to_string(self, pad):
        """Return a padded string representation."""
        s = pad + self.name + ':R^' + str(self.num_input) + ' --> '
        s += 'R^' + str(self.num_output)
        if self.is_function_wrapper:
            s += ' (function wrapper)'
        return s

    def __str__(self):
        """Return a string representation of this object."""
        return self._to_string('')

    def join(self, func):
        """Join the outputs of two functions."""
        func = self._to_func(func)
        functions = (self, )
        if isinstance(self, FunctionJoinedOutputs):
            functions = self.functions
        functions += (func, )
        return FunctionJoinedOutputs(functions)


class _FunctionCollection(Function):

    """A collection of functions."""

    # The functions (a tuple)
    _functions = None

    @property
    def functions(self):
        """Get the functions involved in the summation."""
        return self._functions

    def __init__(self, functions, name='Function Collection'):
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
        for f in functions[1:]:
            if num_input != f.num_input or num_output != f.num_output:
                raise ValueError(
                    'All functions must have the same dimensions.')
        self._functions = functions
        super(_FunctionCollection, self).__init__(num_input, num_output, name=name)

    def _to_string(self, pad):
        """Return a string representation with padding."""
        s = super(_FunctionCollection, self)._to_string(pad)
        for f in self.functions:
            s += '\n' + f._to_string(pad + ' ')
        return s

    def _eval_all(self, x, func):
        """Evaluate func for all the functions in the container."""
        return [getattr(f, func)(x) for f in self.functions]


class FunctionSum(_FunctionCollection):

    """Define the sum of functions."""

    def __init__(self, functions, name='Function Sum'):
        """Initialize the object."""
        super(FunctionSum, self).__init__(functions, name=name)

    def __call__(self, x):
        """Evaluate the function."""
        return np.sum(self._eval_all(x, '__call__'), axis=0)

    def d(self, x):
        """Evaluate the derivative of the function."""
        return np.sum(self._eval_all(x, 'd'), axis=0)


class FunctionJoinedOutputs(_FunctionCollection):

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
        for f in functions[1:]:
            if num_input != f.num_input:
                raise ValueError(
                    'All functions must have the same dimensions.')
            num_output += f.num_output
        self._functions = functions
        super(_FunctionCollection, self).__init__(num_input, num_output,
                                                  name=name)

    def __call__(self, x):
        return np.concatenate(self._eval_all(x, '__call__'), axis=-1)

    def d(self, x):
        return np.concatenate(self._eval_all(x, 'd'), axis=-1)


class FunctionMultiplication(_FunctionCollection):

    """Define the multiplication of functions (element wise)"""

    def __init__(self, functions, name='Function Multiplication'):
        """Initialize the object."""
        super(FunctionMultiplication, self).__init__(functions, name=name)

    def __call__(self, x):
        """Evaluate the function at x."""
        return np.prod(self._eval_all(x, '__call__'), axis=0)

    def d(self, x):
        """Evaluate the derivative of the function."""
        val = self._eval_all(x, '__call__')
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

    def _eval(self, x):
        """Evaluate the function at x."""
        return self.const

    def _d_eval(self, x):
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
                raise ValueError('Dimensions of ' + str(functions[i-1]) + ' and '
                                 + str(functions[i]) + ' do not agree.')
        self._functions = functions
        super(FunctionComposition, self).__init__(num_input, num_output, name=name)

    def __call__(self, x):
        """Evaluate the function at x."""
        z = x
        for f in self.functions[-1::-1]:
            z = f(z)
        return z

    def d(self, x):
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
        super(FunctionPower, self).__init__(f.num_input, f.num_output, name=name)

    def _eval(self, x):
        """Evaluate the function at x."""
        return self.function._eval(x) ** self.exponent


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