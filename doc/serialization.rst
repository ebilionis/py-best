Serialization
=============

The good thing since we are working with Python, is that we can trivially
save our models (any model) to binary files and load them in any other
machine. This is achived through `pickle <http://docs.python.org/2/library/pickle.html>`_
or `cPickle <http://docs.python.org/release/2.5/lib/module-cPickle.html>`_.
:mod:`cPickle` is used in exactly the same way
as :mod:`pickle` (you just change the name...), so we will only work
with :mod:`pickle` here. Use :mod:`cPickle` when :mod:`pickle` becomes
too slow or runs out of memory.

Now, assume that we are working with Best and we have an object ``obj``
(this can be a model, a function, etc.) and that we want to save it
to a file. Here is how we do it::

    import pickle
    with open('foo.bin', 'wb') as fd:
        pickle.dump(obj, fd)

Now, when you want to load it, you just do::

    with open('foo.bin', 'rb') as fd:
        obj = pickle.load(fd)