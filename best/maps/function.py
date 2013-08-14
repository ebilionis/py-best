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
        x = np.array(x)
        if (self.num_input == 1 and isinstance(x, np.ndarray)
            and len(x.shape) == 1):
                x = np.atleast_2d(x).T
        else:
            x = np.atleast_2d(x)
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

    def __call__(self, x):
        """Call the object.

        If y is provided, then the Function should write the output
        on y.

        Arguments:
            x       ---     The input variables.

        """
        x = self._fix_x(x)
        size = x.shape[:-1]
        num_dim = x.shape[-1]
        assert num_dim == self.num_input
        num_pt = np.prod(size)
        x = x.reshape((num_pt, num_dim))
        res = np.ndarray((num_pt, self.num_output, ))
        for i in range(num_pt):
            res[i, :] = self._eval(x[i, :])
        if num_pt == 1:
            if self.num_output == 1:
                return res[0, 0]
            return res[0, :]
        return res.reshape(size + (self.num_output, ))

    def _eval(self, x):
        """Evaluate the function assume x has the right dimensions."""
        if self.is_function_wrapper:
            return self.f_wrapped(x)
        else:
            raise NotImplementedError()

    def d(self, x):
        """Evaluate the derivative of the function at x."""
        x = self._fix_x(x)
        size = x.shape[:-1]
        num_dim = x.shape[-1]
        assert num_dim == self.num_input
        num_pt = np.prod(size)
        x = x.reshape((num_pt, num_dim))
        res = np.ndarray((num_pt, self.num_output, self.num_input))
        for i in range(num_pt):
            res[i, :, :] = self._d_eval(x[i, :])
        if num_pt == 1:
            if self.num_output == 1:
                if self.num_input == 1:
                    return res[0, 0, 0]
                return res[0, 0, :]
            if self.num_input == 1:
                return res[0, :, 0]
            return res[0, :, :]
        return res.reshape(size + (self.num_output, self.num_input))

    def _d_eval(self, x):
        """Evaluate the derivative of the function at x.

        It should return a 2D numpy array of dimensions
        num_output x num_input. That is, it should return
        the Jacobian at x.
        """
        raise NotImplementedError()

    def __add__(self, func):
        """Add this function with another one."""
        if isinstance(func, float) or isinstance(func, int):
            func = np.ones(self.num_output) * func
        if isinstance(func, np.ndarray):
            func = ConstantFunction(self.num_input, func)
        if self is func:
            return self * 2
        if not isinstance(func, Function) and hasattr(func, '__call__'):
            func = Function(self.num_input, self.num_output, f_wrapped=func)
        f = FunctionSum((self, func))
        return f

    def __mul__(self, func):
        """Multiply two functions."""
        if isinstance(func, float) or isinstance(func, int):
            func = np.ones(self.num_output) * func
        if isinstance(func, np.ndarray):
            func = ConstantFunction(self.num_input, func)
        if self is func:
            return FunctionPower(self, 2.)
        if not isinstance(func, Function) and hasattr(func, '__call__'):
            func = Function(self.num_input, self.num_output, f_wrapped=func)
        f = FunctionMultiplication((self, func))
        return f

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


class FunctionSum(_FunctionCollection):

    """Define the sum of functions."""

    def __init__(self, functions, name='Function Sum'):
        """Initialize the object."""
        super(FunctionSum, self).__init__(functions, name=name)

    def _eval(self, x):
        """Evaluate the function."""
        y = np.zeros(self.num_output)
        for f in self.functions:
            y += f(x)
        return y


class FunctionMultiplication(_FunctionCollection):

    """Define the multiplication of functions (element wise)"""

    def __init__(self, functions, name='Function Multiplication'):
        """Initialize the object."""
        super(FunctionMultiplication, self).__init__(functions, name=name)

    def _eval(self, x):
        """Evaluate the function at x."""
        y = np.ones(self.num_output)
        for f in self.functions:
            y *= f(x)
        return y


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

    def _eval(self, x):
        """Evaluate the function at x."""
        z = x
        for f in self.functions:
            z = f(z)
        return z

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
            in_idx = range(self.screened_func.num_input)
        elif not (isinstance(in_idx, tuple) or isinstance(in_idx, list)):
            raise TypeError('in_idx must be a tuple or a list.')
        self._in_idx = in_idx
        if default_inputs is None:
            default_inputs = np.zeros(self.screened_func.num_input)
        elif not isinstance(default_inputs, np.ndarray):
            raise TypeError('default_inputs must be a numpy array.')
        if not len(default_inputs.shape) == 1:
            raise TypeError('default_inputs must be 1D.')
        elif not default_inputs.shape[0] == self.screened_func.num_input:
            raise TypeError('default_inputs must have the same number of '
                            + 'dimensions as screened_func.')
        self._default_inputs = default_inputs
        if out_idx is None:
            out_idx = range(self.screened_func.num_output)
        elif not (isinstance(out_idx, tuple) or isinstance(out_idx, list)):
            raise TypeError('out_idx must be a tuple or a list.')
        self._out_idx = out_idx
        num_input = len(self.in_idx)
        num_output = len(self.out_idx)
        super(FunctionScreened, self).__init__(num_input, num_output,
                                               name=name)

    def _eval(self, x):
        """Evaluate the function at x."""
        x_full = self.default_inputs.copy()
        x_full[self.in_idx] = x
        print 'x: ', x
        print 'x_full: ', x_full
        y_full = self.screened_func(x_full)
        return y_full[self.out_idx]

    def _to_string(self, pad):
        """Return a padded string representation."""
        s = super(FunctionScreened, self)._to_string(pad) + '\n'
        s += self.screened_func._to_string(pad + ' ') + '\n'
        s += pad + ' in_idx: ' + str(self.in_idx) + '\n'
        s += pad + ' default_inputs: ' + str(self.default_inputs) + '\n'
        s += pad + ' out_idx: ' + str(self.out_idx)
        return s