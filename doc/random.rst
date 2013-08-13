Random
======

.. module:: best.random
    :synopsis: Classes and functions related to random number generation.

The purpose of :mod:`best.random` is to define various probabilistic
concepts that are useful in uncertainty quantification tasks as well as
in solving inverse problems. The module is trying to make as much use
as possible of
`scipy.stats <http://docs.scipy.org/doc/scipy/reference/stats.html>`_.
This module intends to be a generalization to random vectors of the
concepts found there. Therefore, it suggested that you are familiar
with ``scipy.stats`` before moving forward.


Conditional Random Variable
---------------------------

The class :class:`best.random.RandomVariableConditional` defines a
1D random variable conditioned to live in a subinterval. Generally
speaking, let :math:`x` be a random variable and :math:`p_x(x)` its
probability density. Now let, :math:`y` be the truncation of :math:`x`
conditioned to live on the interval :math:`[a, b]`. Its probability
density is:

    .. math:: p_y(y) = \frac{p_x(x)1_{[a, b]}(x)}{\int_a^b p_x(x')dx'}.

Here is the definition of this class:

.. class:: best.random.RandomVariableConditional

    A 1D random variable conditioned to live in a subinterval.

    The class inherits :class:`scipy.stats.rv_continuous` and
    overloads the functions ``_pdf`` and ``_cdf``. The rest of the
    functionality is provided automatically by
    :class:`scipy.stats.rv_continuous`.

    .. method:: __init__(random_variable, subinterval)
        Initialize the object.

        :param random_variable: A 1D random variable.
        :type random_variable: :class:`scipy.stats.rv_continuous` or
        :class:`best.random.RandomVariableConditional`
        :param subinterval: The subinterval :math:`[a, b]`.
        :type subinterval: tuple, list or numpy array with two elements

    .. method:: pdf(y)
        Get the probability density at ``y``.

        :param y: The evaluation point.
        :type y: float

    .. method:: cdf(y)
        Get the cummulative distribution function at ``y``.


        This is:

            .. math:: F_y(y) = \int_a^y p_y(y')dy' = \frac{F_x(\min(y, b)) - F_x(a)}{F_x(b) - F_x(a)}.

        :param y: The evaluation point.
        :type y: float

    .. method:: rvs([size=1[, loc=0[, scale=1]]])
        Take samples from the probability density.

        :Note: This is carried out by rejection sampling. Therefore, it \
               can potentially be very slow if the interval is very \
               overload it if you want something faster.

        :Todo: In future editions this could use some MCMC variant.

        :param size: The shape of the samples you want to take.
        :type size: int or tuple/list of integers
        :param loc: Shift the random variable by ``loc``.
        :type loc: float
        :param scale: Scale the random variable by ``scale``.
        :type scale: float
        :returns: The samples

    .. method:: split([pt=None])
        Split the random variable in two conditioned random variables.

        Creates two random variables :math:`y_1` and :math:`y_2` with
        probability densities:

        .. math:: p_{y_1}(y_1) = \frac{p_x(x)1_{[a, \text{pt}]}(x)}{\int_a^{\text{pt}} p_x(x')dx'}

        and

        .. math:: p_{y_2}(y_2) = \frac{p_x(x)1_{[\text{pt}, b]}(x)}{\int_{\text{pt}}^b p_x(x')dx'}.

        :param pt: The splitting point. If None, then the median is used.
        :type pt: float or NoneType
        :returns: A tuple of two :class:`best.random.RandomVariableConditional`

Now, let's look at an example.
Let's create a random variable :math:`x` with an Exponential probability
density:

    .. math:: p(x) = e^{-x},

and construct the conditioned random variable :math:`y` by restricting
:math:`p(x)` on :math:`(1, 2)`::

    import scipy.stats
    import best.random

    px = scipy.stats.expon()
    py = best.random.RandomVariableConditional(px, (1, 2), name='Conditioned Exponential')
    print str(py)
    print py.rvs(size=10)
    print py.interval(0.5)
    print py.median()
    print py.mean()
    print py.var()
    print py.std()
    print py.stats()
    print py.moment(10)

which gives::

    Conditioned Random Variable: Conditioned Exponential
    Original interval: (1, 2)
    Subinterval: (1, 2)
    Prob of sub: 0.232544157935
    [ 1.67915905  1.18775814  1.78365754  1.3167513   1.33650141  1.19931135
      1.85734068  1.74867647  1.35161718  1.55198301]
    (1.1720110607571301, 1.6426259804912111)
    1.37988549304
    1.41802329313
    0.0793264057922
    0.281649437763
    (array(1.4180232931306735), array(0.0793264057922074))
    129.491205116

Here is, how you can visualize the pdf and the cdf::

    import numpy as np
    import matplotlib.pyplot as plt
    y = np.linspace(0, 4, 100)
    plt.plot(y, py.pdf(y), y, py.cdf(y), linewidth=2.)
    plt.legend(['PDF', 'CDF'])
    plt.show()

which gives you the following figure:

    .. figure:: images/rv_conditional.png
        :align: center

        PDF and CDF of a conditioned Exponential random variable.

Now, let's split it in half and visualize the two other random variables::

    py1, py2 = py.split()
    print str(py1)
    print str(py2)

which prints::

    Conditioned Random Variable: Conditioned Random Variable
    Original interval: (0.0, inf)
    Subinterval: [ 1.          1.38629436]
    Prob of sub: 0.117879441171
    Conditioned Random Variable: Conditioned Random Variable
    Original interval: (0.0, inf)
    Subinterval: [ 1.38629436  2.        ]
    Prob of sub: 0.114664716763

and also creates the following figure:

    .. figure:: images/rv_conditional_split.png
        :align: center

        PDF and CDF of the two random variables that occure after splitting
        in two the conditioned Exponential random variable of this example.


Random Vector
-------------

The class :class:`best.random.RandomVector` represents a random vector.
The purpose of this class is to serve as a generalization of
``scipy.stats.rv_continuous``. It should offer pretty much the same
functionality. It should be inherited by all classes that wish to be
random vectors.

Here is a basic reference for the class:

.. class:: best.random.RandomVector

    .. method:: __init__(support[, name='Random Vector')
        Initialize the object.

        The ``support`` is an object representing the support
        of the random vector. It has to be a :class:`best.Domain` or
        a rectangle represented by lists, tuples or numpy arrays.

        :param support: The support of the random variable.
        :type support: :class:`best.Domain` or a rectangle
        :param name: A name for the random vector.
        :type name: str

    .. attribute:: support
        Get the support of the random vector.

    .. attribute:: num_dim
        Get the number of dimensions of the random vector.

    .. attribute:: name
        Get the name of the random vector.

    .. method:: logpdf(x)
        Get the logarithm of the probability density at ``x``.

    .. method:: pdf(x)
        Get the probability density at ``x``.

    .. method:: logcdf(x)
        Get the logarithm of the cummulative distribution function at ``x``.

    .. method:: cdf(x)
        Get the cummulative distribution function at ``x``.

    .. method:: moment(n)
        Get the non-central n-th moment of the random vector.

    .. method:: entropy()
        Get the entropy of the random vector.

    .. method:: mean()
        Get the mean of the random vector.

    .. var:: var()
        Get the variance of the random vector.

    .. std:: std()
        Get the standard deviation of the random vector.