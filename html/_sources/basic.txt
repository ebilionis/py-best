.. _basic:

Basic Objects
=============

There are a few things that every :mod:`best` object needs to do.
Most importantly we would like to be able to recognize that they are
indeede :mod:`best` objects.
In addition, we want them to have a name that can be assinged at will
and to be able to give a string representation of themselves.
These requirements are met by forcing every object in :mod:`best`
to be a child of the class :class:`best.Object`:

.. class:: best.Object

    :inherits: :class:`object`

    A class that all :mod:`best` objects should inherit.

    .. method:: __init__([name='Best Object'])

        Initialize the object.

        :param name: A name for the object.
        :type name: ``str``

    .. attribute:: name

        Get/set the name of the object.

    .. method:: _to_string(pad)

        Return a string representation of the object padding the
        beginning with ``pad``.

        This can and should be overloaded by children.

        :param pad: The padding we will use.
        :type pad: ``str``
        :returns: A string representation of the object.
        :rtype: ``str``

    .. method:: __str__()

        Return a string representation of the object. It is equivalent
        to calling ``self._to_string('')``.