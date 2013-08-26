.. _domain:

Domain
======

.. module:: best.domain
    :synopsis: Represents a domain of a Euclidean space.


The :mod:`best.domain` is used to represent domains of a Euclidean space.
It is used to represent the support of a
:class:`best.random.RandomVector`.


.. class:: best.domain.Domain

    :inherits: :class:`best.Object`

    A class that represents a domain of a Euclidean space.

    .. method:: __init__(num_dim[, name='Domain'])

        Initialize the object.

        :param num_dim: The number of dimensions.
        :type num_dim: int

    .. attribute:: num_dim

        Get the number of dimensions

    .. method:: is_in(x)

        Check if ``x`` is inside the domain.

        :note: This must be implemented by the children of this class.
        :param x: An point.
        :type x: 1D numpy array
        :throws: :class:`NotImplementedError`

    .. method:: _to_string(pad):

        :overloads: :func:`best.Object._to_string()`


.. class:: best.domain.Rectangle

    :inherits: :class:`best.domain.Domain`

    A class that represents a rectangular domain.

    .. method:: __init__(rectangle[name='Rectangular Domain'])

        Initialize the object.

        :param rectangle: This must be a ``num_dim x 2`` array \
                          representing a rectangle.
        :type rectangle: 2D numpy array or list/tuple of list/tuples of \
                         ``float``

    .. attribute:: rectangle

        Get the rectangle.

    .. is_in(x)

        :overloads: :func:`best.domain.Domain.is_in()`

    .. _to_string(pad)

        :overlods: :func:`best.domain.Domain._to_string()`


.. class:: best.domain.UnitCube

    :inherits: :class:`best.domain.Rectangle`

    A class that represents a unit cube domain.

    .. method:: __init__(num_dim[, self='Unit Cube Domain'])

        Initialize the object.

        :param num_dim: The number of dimensions.
        :type num_dim: ``int``


.. class:: best.domain.AllSpace

    :inherits: :class:`best.domain.Rectangle`

    A class that represents all space.

    .. method:: __init__(num_dim[, self='All Space Domain'])

        Initialize the object.

        :param num_dim: The number of dimensions.
        :type num_dim: ``int``