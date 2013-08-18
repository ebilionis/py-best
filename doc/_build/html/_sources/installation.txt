Installation
===========

BEST is mostly written in Python, but there are some parts that written
in C/C++ and we also interface some Fortran codes. There is a small
part that needs to be compiled.


Dependences
-----------

BEST requires the following libraries:
    * `Numpy <http://www.numpy.org/>`_
    * `Scipy <http://www.scipy.org/>`_
    * `LAPACK <http://www.netlib.org/lapack/>`_
    * `Boost C++ Libraries <http://www.boost.org/>`_ with
      ``libboost-python``.

We need access to LAPACK because we are using some factorizations that
are not interfaced by Numpy or Scipy.

The following libraries are recommened if you want to use some of the
functionality of BEST:
    * `mpi4py <http://mpi4py.scipy.org/>`_ for parallel computing.
    * `matplotlib <http://matplotlib.org/>`_ for plotting.

These need to be installed before attempting to isntall BEST itself.


Compiling
---------

Everything that needs to be compiled is in ``py-best/src``.
You should define the following variables::

    $ export BOOST_DIR=/path/to/boost/instalation
    $ export PYTHON_INC_DIR=/path/to/python/include/directory
    $ export NUMPY_INC_DIR=/path/to/numpy/include/directory
    $ export LAPACK_LIB_DIR=/path/to/lapack/library

If you need to link against vendor specific LAPACK, simply edit
``py-best/src/build.sh``. The building process is trivial to understand.
Once, everything is set all you have to do is::

    $ cd py-best/src
    $ ./build.sh


Configuring your Python Environment
-----------------------------------

All you have to do is configure your ``$PYTHON_PATH`` to point
to ``py-best``. Something like::

    export PYTHON_PATH=/path/to/py-best:$PYTHON_PATH