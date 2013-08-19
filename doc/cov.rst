.. _cov:

Covariance Functions
====================

Covariance functions are a special choice of a multi-input
single-output functions that are treated separately from common
functions. A covariance function is a function:

    .. math:: k:\mathbb{R}^d\times\mathbb{R}^d\rightarrow\mathbb{R},

that is **semi-positive definite**. That is, for all
:math:`\mathbf{x}^{(i)}` we have:

    .. math::
        \sum_i\sum_j
            k\left(\mathbf{x}^{(i)}, \mathbf{x}^{(j)}\right) \ge 0.

The :class:`best.maps.CovarianceFunction` class
-----------------------------------------------

All covariance functions must inherit from :class:`best.maps.CovarianceFunction`:

.. class:: best.maps.CovarianceFunction

    A class representing a covariance function. This is any
    funciton :math:`k(\mathbf{x}, \mathbf{x}'; \boldsymbol{\theta})`.
    The parameters :math:`\boldsymbol{\theta}` are parameters
    of the covariance function that can be set. Within the class
    documentation, we call them ``hyp`` (from hyper-parameters).
    The hyper-parameters should always be represented by a 1D
    numpy array.

    .. method:: __init__(num_input[, num_hyp=0[, hyp=None[, \
                         k_wrapped=None[, \
                         name='Covariance Function']]]])

        Initialize the object.

        :param num_input: The dimensionality of the input points.
        :type num_input: int
        :param num_hyp: The number of hyper-parameters of the covariance \
                        function. The default value is zero.
        :type num_hyp: int
        :param hyp: A vector of hyper-parameters. If None, then the \
                    object stores no hyper-parameters and cannot \
                    be evaluated without providing a value for them. \
                    If set to a valid hyper-parameter object, \
                    then the object can be evaluated without any \
                    additional info.
        :type hyp: 1D numpy array
        :param k_wrapped: A common python function defined by \
                          of two variables returning a real number. \
                          Ignored if None.
        :param name: A name for the covariance function.
        :type name: str

    .. attribute:: num_input

        Get the dimensionality of the input.

    .. attribute:: name

        Get the name.

    .. attribute:: k_wrapped

        Get the wrapped function.

    .. attribute:: is_wrapped

        Return ``True`` if there is a wrapped function.

    .. attribute:: hyp

        Get/Set the hyper-parameters. If the hyper-parameters
        have not been set, then this returns ``None``.
        You might want to check
        :func:`best.maps.CovarianceFunction.is_hyp_set()` before
        calling this one.

    .. attribute:: num_hyp

        The number of hyper-parameters.

    .. method:: __call__(x, y[, hyp=None])

        Evaluate the covariance function at ``x`` and ``y``.

        :Note: If ``hyp`` has already been set, then you do not have \
               provide. If you do, then the value you have set should \
               be ignored. If it is not set and you pass ``None``, then \
               it should throw a ``ValueError()``.

        :param x: Input points for ``x``.
        :type x: 1D or 2D numpy array.
        :param y: Input points for ``y``.
        :type y: 1D or 2D numpy array.
        :param hyp: The hyper-parameters.
        :type hyp: 1D array.

        The return type depends on what ``x`` and ``y`` are.
        Let :math:`n, m` be
        the dimension of ``x`` and ``y``, respectively. Then the function
        should return a 2D numpy array with shape :math:`n \times m` with:

        .. math::

            K_{ij} = k\left(\mathbf{x}^{(i)}, \mathbf{y}^{(j)}; \boldsymbol{\theta}\right).

        This is known as the **cross covariance matrix** of ``x`` and ``y``.
        If ``x`` and ``y`` are the same, then it is semi-positive definite
        and it is known as the **covariance matrix** of ``x``.

        .. warning::

            You do not have to overload this function when creating
            your own covariance function. However, doing so might
            yield a big performance gain. When you do so, you must
            conform with the types that must be returned for each
            special case of ``x`` and ``y``.

    .. method:: d(x, y[, hyp=None])

        Evaluate the Jacobian of the cross covariance matrix
        with respect to the inputs.

        Parameters as in
        :func:`best.maps.CovarianceFunction.__call__()`.

        The return type depends on ``x`` and ``y``. Let :math:`n, m`
        be as before anb assume that the dimensionality of
        the input is :math:`d`. Then, this should
        return a 3D numpy array with shape :math:`n \times m \times d`
        containing:

        .. math::

            D_{ijl} = \frac{\partial k\left(\mathbf{x}^{(i)}, \mathbf{y}^{(j)}; \boldsymbol{\theta}\right)}{\partial y_l}.

        The same warning as in :func:`best.maps.CovarianceFunction.__call__()`
        applies here in case you choose to overload it.

    .. method:: d_hyp(x, y[, hyp=None])

        Evaluate the Jacobian of the cross covariance matrix
        with respect to the hyper-parameters.

        Parameters as in
        :func:`best.maps.CovarianceFunction.__call__()`.

        The return type depends on ``x`` and ``y``. Let :math:`n, m`
        be as before anb assume that the dimensionality of
        the hyper-parameters is :math:`q`. Then, this should
        return a 3D numpy array with shape :math:`n \times m \times q`
        containing:

        .. math::

            E_{ijl} = \frac{\partial k\left(\mathbf{x}^{(i)}, \mathbf{y}^{(j)}; \boldsymbol{\theta}\right)}{\partial \theta_l}.

        The same warning as in :func:`best.maps.CovarianceFunction.__call__()`
        applies here in case you choose to overload it.

    .. method:: _eval(x, y, hyp)

        Evaluate the covariance function at two single inputs given ``hyp``.
        This is the function that you can overload to access to
        the calculation of the cross covariance matrix.

    .. method:: _d_eval(x, y, hyp)

        Evaluate the Jacobian of the covariance function at two single
        inputs given ``hyp``. The result should be a 1D numpy
        array of ``num_input`` elements. You can overloaded this
        to gain access to
        :func:`best.maps.CovarianceFunction.d()`.

    .. method:: _d_hyp_eval(x, y, hyp)

        Evaluate the Jacobian of the covariance function at two
        single inputs given ``hyp``. The result should be a 1D numpy
        array of ``num_hyp`` elements. You can overload this to gain
        access to
        :func:`best.maps.CovarianceFunction.d_hyp()`.

    .. method:: __str__()

        Return a string representation of the object.

    .. method:: __mul__(g)

        Return a new covariance function that is the product of the
        current one and ``g``.

        :param g: A covariance function.
        :type g: :class:`best.maps.CovarianceFunction`

    .. method:: __add__(g)

        Return a new covariance function that is the sum of the
        current one and ``g``.

        :param g: A covariance function.
        :type g: :class:`best.maps.CovarianceFunction`

    .. method:: to_basis(X)

        Return a basis object from a covariance function.

        The parameters are as in
        :class:`best.maps.CovarianceFunctionBasis`.
        See the documentation there for more details.


.. _cov-example:

Examples of Covariance Functions
--------------------------------

.. _cov-se:

Squared Exponential Covariance
++++++++++++++++++++++++++++++

The class :class:`best.maps.CovarianceFunctionSE` implements a
particular case of a :class:`best.maps.CovarianceFunction`:
the **Squared Exponential** (SE) covariance function:

.. math::
    k\left(\mathbf{x}, \mathbf{y}; \boldsymbol{\theta}
    \right) =
    \exp\left\{
        -\frac{1}{2}
        \frac{\sum_{i=1}^d\left(x_i - y_i\right)^2}{\theta_i^2}
    \right\}.

Let's plot it in 1D::

    import numpy as np
    import matplotlib.pyplot as plt
    from best.maps import CovarianceFunctionSE
    k = CovarianceFunctionSE(1)
    x = np.linspace(-5, 5, 100)
    plt.plot(x, k(0, x, hyp=1).T)

You should see the following:

    .. figure:: images/cov_se.png
        :align: center

        Plot of :math:`k(0, x; 1)`.

Here is how you can get the derivative with respect to the input::

    D = k.d(0, x, hyp=1.)
    plt.plot(x, D[:, :, 0].T)
    plt.show()

You should see:

    .. figure:: images/cov_se_der.png
        :align: center

        Plot of :math:`\frac{\partial k(0, x; 1)}{\partial x}`.

Here is how you can get the derivative with respect to the hyper-parameters::

    E = k.d_hyp(0, x, hyp=1.)


.. _cov-basis:

Constructing Basis from Covariance Functions
--------------------------------------------

Given a covariance function :math:`k(\cdot, \cdot; \boldsymbol{\theta})`, we
can construct a basis to be used in generalized linear models
(see :class:`best.maps.GeneralizedLinearModel`).
All we need is a set of input points
:math:`\left\{\mathbf{x}^{(i))}\right\}_{i=1}^n` and we can get
a set of basis functions
:math:`\left\{\phi_i(\cdot; \boldsymbol{\theta})\right\}_{i=1}^n`
Here is how:

    .. math::
        \phi_i(\mathbf{x}; \boldsymbol{\theta}) =
        k(\mathbf{x}^{(i)}, \mathbf{x}; \boldsymbol{\theta}).

If the covariance function depends only on the distance between
:math:`x` and :math:`y`, it is known
as `Radial Basis Function <http://en.wikipedia.org/wiki/Radial_basis_function>`_ (RBF).
Usually the :math:`\left\{\mathbf{x}^{(i))}\right\}_{i=1}^n` are the
observed input points. Such a basis can be used to train a
:class:`best.rvm.RelevanceVectorMachine` or other types of
:class:`best.maps.GeneralizedLinearModel`.

It is quite simple to construct this basis in Best.
Let ``X`` be a 2D array of input points and ``k`` a covariance function.
Then, you can construct the basis by::

    phi = k.to_basis(X, hyp=1.)

Here is how this looks in the previous example for
a random selection of 10 input points:

    .. figure:: images/cov_se_basis.png
        :align: center

        Plot of the :math:`\phi_i(x)` for the 1D example.

This functionality is offered via the following class:

.. class:: best.maps.CovarianceFunctionBasis

    Represents a basis constructed from a covariance function.
    The class inherits (as every basis) from
    :class:`best.maps.Function`. So, there is no need to give here
    the complete documentation. Simply, use a function!

    .. method:: __init__(k, X[, hyp=None])

        Initializes the object.

        :param k: A covariance function.
        :type k: :class:`best.maps.CovarianceFunction`
        :param X: A collection of input points that serve as centers.
        :type X: 2D numpy array
        :param hyp: A set of hyper-parameters that will remain fixed. \
                    If ``None``, then we will copy the parameters that \
                    are already in ``k``. We will throw an exception \
                    if we find nothing there.
        :type hyp: 1D numpy array

    .. attribute:: hyp

        Get the hyper-parameters.

    .. attribute:: cov

        Get the underlying covariance function.

    .. attribute:: X

        Get the centers.