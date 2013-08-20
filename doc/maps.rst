.. _map:

Maps
====

.. module:: best.maps
    :synopsis: Generic definition of multi-input/output functions and\
               their allowed operations.


The purpose of the :mod:`best.maps` module is to define generic manner the
concept of a multi-input/output function. Our goal is to have function
objects that can be combined easily in arbitrary ways to create new
functions. The complete details of the implementation can be found in the
docstrings.


.. _map-basic:

The basic concepts
------------------

We will say that an object is a *regular multi-input/output function* if
it is a common python function that accepts as input a unique numpy array
and returns a unique numpy array. For example::

    def f(x):
        assert isinstance(x, np.ndarray)
        assert len(x.shape) == 1
        return x + x

is a *regular multi-input/output function*.

However, we will give a stronger definition for what we are going to call
a, simply, a *multi-input/output function*.
So, by a *multi-input/output function* we refer to any
object that inherits from the class :class:`best.maps.Function`.
:class:`best.maps.Functions` have be designed with the following points in
mind:

    * It needs to know the number of input/output dimensions.
    * It needs to have a name.
    * It should have a user-friendly string representation.
    * It can represent a *regular multi-input/output function* as defined \
      above.
    * If the dimensions of two multi-input/output functions are \
      compatible, then it should be easy to combine them in order to \
      construct linear combinations of them.

The most important idea is that anything in Best that can be
thought as such a map, must be a child of :class:`best.maps.Function`.

The :class:`best.maps.Function` class.
--------------------------------------

We will not dwelve into the implementational details of
:class:`best.maps.Function`. Our goal here is to document its functions
that should/could be overloaded by its children and then demonstrate by
a simple example what is the functionality it actually provides.

.. class:: best.maps.Function

    :inherits: :class:`best.Object`

    A class representing an arbitrary multi-input/output function.

    Everything in Best that can be though as such a function should be a
    child of this class.

    .. method:: __init__(num_input, num_output[, num_hyp=0[, \
                         hyp=None[, name="function"[, \
                         f_wrapped=None]]]])

        Initializes the object.

        You do not have to overload this method. However, if you choose
        to do so, keep in mind that children of this class should call
        this constructor providing at least the first two arguments.
        This can be achieved with the following code::

            class MyFunction(best.maps.Function):

                def __init__(self):
                    # Assuming here that we now the number of inputs and
                    # outputs of the function (as well as its name).
                    super(MyName, self).__init__(10, 24, name='Super Function')

        :param num_input: The number of input dimensions.
        :type num_input: ``int``
        :param num_output: The number of output dimensions.
        :type num_output: ``int``
        :param num_hyp: The number of hyper-parameters of the function. \
                        Zero by default.
        :type num_hyp: ``int``
        :param hyp: The hyper-parameters of the function. If ``None`` \
                    they are left unspecified and must be provided \
                    when calling it.
        :type hyp: 1D numpy array
        :param name: A name for the function you create.
        :type name: ``str``
        :param f_wrapped: If specified, then this function is simply a \
                          f_wrapped.
        :type f_wrapped: A normal python function that is a \
                         multi-input/output function.

        You would typically use the last option, in order to construct
        a :class:`best.maps.Function` out of an existing
        regular multi-input/output function. Assuming you have a function
        :func:`f()` (as the one we defined above), then here is how you
        can actually do it::

            F = best.maps.Functions(10, 20, f_wrapped=f)

        where, of course, we have assumed that :func:`f()` accepts a numpy
        array with 10 dimensions and response with one with 20.

    .. attribute:: num_input

        Get the number of input dimensions.

        It cannot be changed directly.

    .. attribute:: num_output

        Get the number of output dimensions.

        It cannot be changed directly.

    .. attribute:: num_hyp

        The number of hyper-parameters.

    .. attribute:: hyp

        Get/Set the hyper-parameters.

    .. attribute:: f_wrapped

        Get the wrapped function (if any).

        It cannot be changed directly.

    .. attribute:: is_function_wrapper

        True if the object is a function wrapper, False otherwise.

    .. method:: _eval(x, hyp):

        Evaluate the function at ``x`` assuming that ``x`` has the
        right dimensions.

        .. note:: This must be re-implemented by children.

        :param x: The evaluation point.
        :type x: 1D numpy array of the right dimensions
        :param hyp: The hyper-parameters. Ignore it if your function \
                    does not have any.
        :type hyp: 1D numpy array.
        :returns: The result.
        :rtype: 1D numpy array of the right dimensions or just a float
        :etype: NotImplementedError

    .. method:: __call__(x[, hyp=None]):

        Evaluate the function ``x``.

        .. note:: This calls :func:`best.maps.Function._eval()`.

        :param x: The evaluation point(s).
        :type x: Can be a multi-dimensional numpy array whose last
                 dimension corresponds to the number of inputs while
                 the rest simply correspond to different evaluation
                 points.
        :param hyp: The hyper-parameters. Ignore it if your function \
                    does not have any.
        :type hyp: 1D numpy array.
        :returns y: The result.
        :rtype: a numpy array of the right dimensions.

    .. method:: _d_eval(x, hyp):

        Evaluate the Jacobian of the function at ``x``. The dimensions
        of the Jacobian are ``num_output x num_input``.

        .. note:: This must be re-implemented by children.

        :param x: The evaluation point.
        :type x: 1D numpy array of the right dimensions
        :param hyp: The hyper-parameters. Ignore it if your function \
                    does not have any.
        :type hyp: 1D numpy array.
        :returns: The Jacobian at ``x``.
        :rtype: 2D numpy array of the right dimensions

    .. method:: d(x[, hyp=None])

        Evaluate the Jacobian of the function at ``x``.

        .. note:: This calls :func:`best.maps.Function._d_eval()`.

        :param x: The evaluation point(s).
        :type x: Can be a multi-dimensional numpy array whose last
                 dimension corresponds to the number of inputs while
                 the rest simply correspond to different evaluation
                 points.
        :param hyp: The hyper-parameters. Ignore it if your function \
                    does not have any.
        :type hyp: 1D numpy array.
        :returns y: The result.
        :rtype: a numpy array of the right dimensions.

    .. method:: d_hyp(x[, hyp=None])

        Evaluate the Jacobian of the function at ``x`` with respect to
        the hyper-parameters.

        .. note:: This calls :func:`best.maps.Function._d_hyp_eval()`.

        :param x: The evaluation point(s).
        :type x: Can be a multi-dimensional numpy array whose last
                 dimension corresponds to the number of inputs while
                 the rest simply correspond to different evaluation
                 points.
        :param hyp: The hyper-parameters. Ignore it if your function \
                    does not have any.
        :type hyp: 1D numpy array.
        :returns y: The result.
        :rtype: a numpy array of the right dimensions.

    .. method:: __add__(g):

        Add two functions.

        :param g: A function to be added to the current object.
        :type g: :class:`best.maps.Function` object, regular \
                 multi-input/output function or just a number.
        :returns: A function object that represents the addition of the
                  current object and ``g``.
        :rtype: :class:`best.maps.Function`

    .. method:: __mul__(g):

        Multiply two functions.

        :param g: A function to be multiplied with the current object.
        :type g: :class:`best.maps.Function` object, regular \
                 multi-input/output function or just a number.
        :returns: A function object that represents the multiplication of \
                  the current object and ``g``.
        :rtype: :class:`best.maps.Function`

    .. method:: compose(g):

        Compose two functions.

        :param g: A function whose output has the same dimensions as the \
                  input of the current object.
        :type g: :class:`best.maps.Function`
        :returns: A function object that represents the composition of \
                  the current object and ``g``.
        :rtype: :class:`best.maps.Function`

    .. method:: join(g):

        Joins the outputs of two functions.

        :param g: A function whose output has the same dimensions as the \
                  input of the current object.
        :type g: :class:`best.maps.Function`
        :returns: A function object that represents jointly the outputs \
                  of the current object and ``g`` (fist ``f`` then ``g``).
        :rtype: :class:`best.maps.Function`

    .. method:: screen([in_idx=None[, out_idx=None[, \
                        default_inputs=None[, name='Screened Function']]]])

        Creates a screened version of the function.

        The parameters are
        as in the constructor of :class:`best.maps.FunctionScreened`.
        You may consult it for details.

    .. method:: _to_string(pad):

        :overloads: :func:`best.Object._to_string()`


.. _map-examples:

Some Examples
-------------
The first example we consider is creating a :class:`best.maps.Function`
wrapper of a regular multi-input/output function::

        import best.maps

        def f(x):
            return x + x

        ff = best.maps.Function(10, 10, name='foo', f_wrapped=f)
        print str(ff)
        x = np.random.randn(10)
        print 'Eval at', x
        print ff(x)

If you wish, you may create a new class that inherits
:class:`best.maps.Function`. You are required to overload
:func:`best.maps.Function.__call__()`::

    from best.maps import Function

    class MyFunction(Function):

        def __call__(self, x):
            return x ** 2

Now, assume that we have two functions with the same number of inputs
and outputs :math:`f(\cdot)` and :math:`g(\cdot)`. Let also :math:`c`
be any floating point number. You may now define several functions:

    * Sum of functions :math:`h(\cdot) = f(\cdot) + g(\cdot)`::

        h = f + g

    * Sum of function with a constant :math:`h(\cdot) = f(\cdot) + c`::

        h = f + c

      :Note: The constant must always be on the right side of the operator.

    * Product of functions :math:`h(\cdot) = f(\cdot)g(\cdot)`::

        h = f * g

    * Product of function with a constant :math:`h(\cdot) = f(\cdot)c`::

        h = f * c

      :Note: The constant must always be on the right side of the operator.

Assume that the two functions have compatible dimensions so
that they can be composed (e.g., the number of outputs of
:math:`g(\cdot)` is the same as the number of inputs of :math:`f(\cdot)`.
Then, you can define :math:`h(\cdot) = f(g(\cdot)` by::

    from best.maps import FunctionComposition
    h = FunctionComposition((f, g))

It is also possible to raise a function to a particular power.
For example, the following code defines :math:`h(\cdot) = f(\cdot)^2`::

    from best.maps import FunctionPower
    h = FunctionPower(f, 2.)


.. _map-screened:

Screened Function
-----------------
A very useful class is the :class:`best.maps.FunctionScreened`. It
implements a screened version of another class. We give a brief
discreption of its functionality.

.. class:: best.maps.FunctionScreened

    :inherits: :class:`best.maps.Function`

    A function that serves as a screened version of another function.

    It is useful in applications when you want to fix certain inputs
    to given values and play with the rest and/or if you want to screen
    certain outputs. It is one of the basic building blocks for
    representing the High-Dimensional Representation (HDMR) of a function.

    .. method:: __init__(screened_function[, in_idx=None[, default_inputs=None[, \
                             out_idx=None[, name='Screened Function']]]])

        Initialize the object.

        :param screened_func: The function to be screened.
        :type screened_func: :class:`best.maps.Function`
        :param in_idx: The input indices that are not screened.
                       It must be a valid container.
                       If None, then no inputs are screened. If
                       a non-empty list is suplied, then the
                       argument default_inputs must be suplied.
        :type in_idx: tuple, list or NoneType
        :param default_inputs: If in_idx is not None, then this can be
                               suplied. It is a default set of inputs
                               of the same size as the original input
                               of the screened_func. If it is not
                               given, then it is automatically set to
                               zero. These values will be used to fill
                               in the missing values of the screened
                               inputs.
        :type default_inputs: 1D numpy array
        :param out_idx: The output indices that are not screened.
                        If None, then no output is screened.
        :type out_idx: tuple, list or NoneType
        :param name: A name for the function.
        :type name: str

    .. method:: __call__(x[, hyp=None])

        :overloads: :func:`best.maps.Function.__call__()`

    .. method:: d(x[, hyp=None])

        :overloads: :func:`best.maps.Function.d()`

    .. method:: d_hyp(x[, hyp=None])

        :overloads: :func:`best.maps.Function.d_hyp()`

Let us give a simple example of how it is to be used. Suppose that you
have a function :math:`f(\cdot)` that takes 10 inputs and responds with
10 outputs. Assume that you wish to fix all the inputs to 0.5 with the
exception of the first one and the fifth one and that you only want to
look at the fourth and the sixth outputs. Here is how you can achieve
this using the :class:`best.maps.FunctionScreened`::

    from best.maps import FunctionScreened
    h = FunctionScreened(f, in_idx=[0, 4],
                         default_inputs=np.ones(f.num_input) * .5,
                         out_idx=[[3, 5]])
    print 'Evaluate h at x = [0.3, -1.]:'
    print h(np.array([0.3, -1.]))
    print 'It should be equivalent to evaluating: '
    x_full = np.ones(f.num_input) * .5
    x_full[[0, 4]] = np.array([0.3, -1.])
    print f(x_full)[[3, 5]]


.. _map-basis:

Basis
-----

A **basis** is simply a collection of multi-input functions
:math:`\phi_i(\cdot)`. Therefore, it can be represented by a child of
:class:`best.maps.Function`. We offer several basis functions.
In particular, Orthogonal Polynomials can be constructed using the
functionality of :mod:`best.gpc`. Furthermore, bases can be constructed
from Radial Basis Functions using :func:`best.maps.RadialBasisFunction.to_basis()`.
You should go through the corresponding documentation. Here, we will
simply state a few examples that exploit the functionality of
:mod:`best.maps`.


.. _map-basis-join:

Joining two bases
+++++++++++++++++

Assume that we are given two bases, say ``phi`` and ``psi``.
We can create a basis that contains the bases functions
of ``phi`` and ``psi`` simultaneously by
using the function :meth:`best.maps.Function.join()`::

    phipsi = phi.join(psi)
    print str(phipsi)


.. _map-basis-sparse:

Getting rid of some of the basis functions
++++++++++++++++++++++++++++++++++++++++++
Now, assume that we are given a basis ``phi`` and that we want to
construct an other one that contains only the first and the fifth
basis functions of ``phi``. We can do this as follows::

    sparse_phi = phi.screen(out_idx=[0, 2])
    print str(sparse_phi)