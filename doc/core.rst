.. _core:

Core
====

.. module:: best.core
    :synopsis: Several C++/C and Fortran libraries.

The module :mod:`best.core` contains several C++/C and Fortran libraries
that are interfaced via Python. We provide a brief description of this
functionality here. The functions in :mod:`best.core` are used internally
by the rest of the modules in :mod:`best`.


.. function:: _lhs()

    Latin hyper-cube sampling.

    This is an interface to John Burkardt's
    `latin_center <http://people.sc.fsu.edu/~jburkardt/cpp_src/latin_center/latin_center.html>`_
    written in C++.

    You should probably use the high-level class
    :class:`best.random.LatinHyperCubeDesign`.


.. function:: _ggsvd(jobU, jobV, jobQ, kl, A, B, alpha, beta, U, V, Q, \
                     work, iwork)

    An interface to the LAPACK Fortan routine
    `dggsvd <http://www.netlib.no/netlib/lapack/double/dggsvd.f>`_
    which performs a
    `Generalized Singular Value Decomposition \
    <http://en.wikipedia.org/wiki/Generalized_singular_value_decomposition>`_.

    You should probably use the high-level class
    :class:`best.linalg.GeneralizedSVD`.

.. function:: _pstrf()

    An interface to the LAPACK Fortran routine
    `dpstrf <http://www.netlib.org/lapack/explore-html/dd/dad/dpstrf_8f.html>`_
    which performs
    an holesky factorization with complete
    pivoting of a real symmetric positive semidefinite matrix A.

    You should probably use the high-level class
    :class:`best.linalg.IncompleteCholesky`.