Maps
====

.. module:: best.maps
    :synopsis: Generic definition of multi-input/output functions and\
               their allowed operations.


The purpose of the :mod:`best.maps` module is to define generic manner the
concept of a multi-input/output function. Our goal is to have function
objects that can be combined easily in arbitrary ways to create new
functions.

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
    A class representing an arbitrary multi-input/output function.

    Everything in Best that can be though as such a function should be a
    child of this class.

    .. method:: __init__(num_input, num_output, [name="function", \
                        [f_wrapped=None]])
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
        :type num_input: integer
        :param num_output: The number of output dimensions.
        :type num_output: integer
        :param name: A name for the function you create.
        :type name: string
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

    .. attribute:: name
        Get the name of the function.

        It cannot be changed directly.

    .. attribute:: f_wrapped
        Get the wrapped function (if any).

        It cannot be changed directly.

    .. attribute:: is_function_wrapper
        True if the object is a function wrapper, False otherwise.

    .. method:: __call__(x):
        Evaluate the function.

        .. note::
            This method has to be reimplemented by all children.

        :param x: The input.
        :type x: 1D numpy array.
        :param y: The output.
        :type y: 1D numpy array.
        :etype: NotImplementedError

    .. method:: __add__(g):
        Add two functions.

        :param g: A function to be added to the current object.
        :type g: :class:`best.maps.Function` object, regular \
                 multi-input/output function or just a number.
        :returns: A function object that represents the addition of the
                  current object and g.
        :rtype: :class:`best.maps.Function`

    .. method:: __mul__(g):
        Multiply two functions.

        :param g: A function to be multiplied with the current object.
        :type g: :class:`best.maps.Function` object, regular \
                 multi-input/output function or just a number.
        :returns: A function object that represents the multiplication of \
                  the current object and g.
        :rtype: :class:`best.maps.Function`

    .. method:: compose(g):
        Compose two functions.

        :param g: A function whose output has the same dimensions as the \
                  input of the current object.
        :type g: :class:`best.maps.Function
        :returns: A function object that represents the composition of \
                  the current object and g.
        :rtype: :class:`best.maps.Function`

    .. method:: __str__():
        Return a string representation of the object.

    .. method:: _to_string(pad):
        Return a string representation of the object with padding.

        This may be reimplemented by children classes.