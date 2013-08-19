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

    .. method:: __init__(random_variable, subinterval[, name='Conditional Random Variable'])

        Initialize the object.

        :param random_variable: A 1D random variable.
        :type random_variable: :class:`scipy.stats.rv_continuous` or \
                              :class:`best.random.RandomVariableConditional`
        :param subinterval: The subinterval :math:`[a, b]`.
        :type subinterval: tuple, list or numpy array with two elements
        :param name: A name for the random variable.
        :type name: str

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

    .. method:: rvs([size=1])

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

:Note: In addition to the few methods defined above, \
       :class:`best.random.RandomVariableConditional` has the full
       functionality of
       `scipy.stats.rv_continuous \
       <http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html#scipy.stats.rv_continuous>`_.

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
``scipy.stats.rv_continuous``.
It should be inherited by all classes that wish to be
random vectors.

Here is a basic reference for the class:

.. class:: best.random.RandomVector

    .. method:: __init__(support[, num_input=None[, hyp=None[, \
                         name='Random Vector']]])

        Initialize the object.

        The class inherits from :class:`best.maps.Function`. The
        motivation for this choice is that the mathematical
        definition of a
        `random variable <http://en.wikipedia.org/wiki/Random_variable>`_
        which states that it is a measurable function.
        Now, the inputs of a :class:`best.random.RandomVector` can
        be thought thought as an other random vector given which
        this random vector has a given value. This becomes useful
        in classes that inherit from this one, e.g.
        :class:`best.random.KarhuneLoeveExpansion`.

        The ``support`` is an object representing the support
        of the random vector. It has to be a :class:`best.domain.Domain`
        or a rectangle represented by lists, tuples or numpy arrays.

        :param support: The support of the random variable.
        :type support: :class:`best.domain.Domain` or a rectangle
        :param num_input: The number of inputs. If `None`, then it is
                          set equal to ``support.num_dim``.
        :type num_input: int
        :param num_hyp: The number of hyper-paramers (zero by default).
        :type num_hyp: int
        :param hyp: The hyper-parameters.
        :type hyp: 1D numpy array or ``None``
        :param name: A name for the random vector.
        :type name: str

    .. attribute:: support

        Get the support of the random vector.

    .. attribute:: num_dim

        Get the number of dimensions of the random vector.

    .. attribute:: name

        Get the name of the random vector.

    .. method:: _pdf(x)

        This should return the probability density at ``x`` assuming
        that ``x`` is inside the domain. **It must be implemented by all
        deriving classes.**

        :param x: The evaluation point.
        :type x: 1D numpy array of dimension ``num_dim``

    .. method:: pdf(x)

        Get the probability density at ``x``. It uses
        :func:`besr.random.RandomVector._pdf()`.

        :param x: The evaluation point(s).
        :type x: 1D numpy array of dimension ``num_dim`` or a 2D \
                 numpy array of dimenson ``N x num_dim``

    .. method:: _rvs()

        Get a sample of the random variable. **It must be implemented
        by all deriving classes.**

        :returns: A sample of the random variable.
        :rtype: 1D numpy array if dimension ``num_dim``

    .. method:: rvs([size=1])

        Get many samples from the random variable.

        :param size: The shape of the samples you wish to draw.
        :type size: list or tuple of ints
        :returns: Many samples of the random variable.
        :rtype: numpy array of dimension ``size + (num_dim, )``

    .. method:: moment(n)

        Get the ``n``-th non-centered moment of the random variable.
        This must be implemented by deriving methods.

        :param n: The order of the moment.
        :type n: int
        :returns: The ``n``-th non-centered moment of the random variable.

    .. method:: mean()

        Get the mean of the random variable.

    .. method:: var()

        Get the variance of the random variable.

    .. method:: std()

        Get the standard deviation of the random variable.

    .. method:: stats()

        Get the mean, variance, skewness and kurtosis of the random variable.

    .. method:: expect([func=None[, args=()]])

        Get the expectation of a function with respect to the probability
        density of the random variable. This must be implemented by the
        deriving classes.


Random Vector of Independent Variables
--------------------------------------

The class :class:`best.random.RandomVectorIndependent` represents
a random vector of independent random variables. It inherits the
functionality of :class:`best.random.RandomVector`.
Here is the reference for this class:

.. class:: best.random.RandomVectorIndependent

    A class representing a random vector with independent components.

    .. method:: __init__(components[, name='Independent Random Vector')

        Initialize the object given a container of 1D random variables.

        :param components: A container of 1D random variables.
        :type components: tuple or list of ``scipy.stats.rv_continuous``
        :param name: A name for the randomv vector.
        :type name: str

    .. attribute:: component

        Return the container of the 1D random variables.

    .. method:: __getitem__(i)

        Allows the usage of `[]` operator in order to get access to
        the underlying 1D random variables.

    .. method:: _pdf(x)

        Evaluate the probability density at ``x``. This is an
        overloaded version of :func:`best.random.RandomVector._pdf()`.

    .. method:: _rvs()

        Take a random sample. This is an overloaded version of
        :func:`best.random.RandomVector._rvs()`.

    .. method:: moment(n)

        Return the n-th non-centered moment. This is an overloaded
        version of :func:`best.random.RandomVector.moment()`.

    .. method:: split(dim[, pt=None])

        Split the random vector in two, perpendicular to dimension ``dim``
        at point ``pt``.

        :param dim: The splitting dimension.
        :type dim: int
        :param pt: The splitting point. If ``None``, then the median \
                   of dimension ``dim`` is used.
        :type pt: float
        :returns: A tuple of two random vectors.
        :rtype: tuple of :class:`best.random.RandomVectorIndependent`.

Here are some examples of how you may use this class::

        comp = (stats.expon(), stats.beta(0.4, 0.8), stats.norm())
        rv = best.random.RandomVectorIndependent(comp)
        print str(rv)
        x = rv.rvs()
        print 'One sample: ', x
        print 'pdf:', rv.pdf(x)
        x = rv.rvs(size=10)
        print '10 samples: ', x
        print 'pdf: ', rv.pdf(x)

This prints::

    Random Vector: Independent Random Vector
    Domain: Rectangular Domain < R^3
    Rectangle: [[  0.  inf]
      [  0.   1.]
      [-inf  inf]]
    pdf of domain: 1.0
    One sample:  [ 0.27583967  0.62299007  1.01218697]
    pdf: 0.104129553451
    10 samples:  [[  2.48588069e+00   5.13373494e-01   2.51959945e+00]
      [  7.18463201e-01   8.03626538e-01  -1.30967423e-01]
      [  3.81458502e-01   1.22199215e-01  -5.47956262e-02]
      [  4.80799826e-01   3.75637813e-02   1.10318554e-02]
      [  4.52448778e-01   2.91548860e-05   8.79078586e-01]
      [  3.03627476e+00   2.35715855e-02  -1.18775141e+00]
      [  3.49253408e-01   8.90061454e-01  -8.93935818e-01]
      [  2.29852363e-02   4.55557385e-04   1.13318738e+00]
      [  2.69130645e-01   2.88083586e-02   7.97967613e-01]
      [  4.18872218e-01   9.97623679e-01  -2.24285728e+00]]
    pdf:  [  1.30325502e-01   2.68022713e-02   1.62999252e-01   3.26470531e-03
       9.98952583e-02   7.43932369e+00   3.29566324e-02   1.62779856e-01
       1.26561530e-01   1.25984246e-03]

Here are some statistics::

    print rv.mean()
    print rv.var()
    print rv.std()
    print rv.stats()

This prints::

    [ 1.          0.33333333  0.        ]
    [ 1.         0.1010101  1.       ]
    [ 1.          0.31782086  1.        ]
    (array([ 1.        ,  0.33333333,  0.        ]), array([ 1.       ,  0.1010101,  1.       ]), array([ 2.        ,  0.65550553,  0.        ]), array([ 23.    ,  11.6225,   2.    ]))

Let us split the random vector perpendicular to the first dimension::

    rv1, rv2 = rv.split(0)
    print str(rv1)
    x = rv1.rvs(size=5)
    print x
    print rv1.pdf(x)
    print rv2.pdf(x)
    print str(rv2)
    print x
    x = rv2.rvs(size=5)
    print rv2.pdf(x)

This prints::

    Random Vector: Independent Random Vector
    Domain: Rectangular Domain < R^3
    Rectangle: [[ 0.          0.69314718]
    [ 0.          1.        ]
    [       -inf         inf]]
    pdf of domain: 0.5
    [[  5.19548316e-01   7.68241112e-01   3.91270986e-01]
     [  1.39697221e-01   1.45666923e-02  -4.77341007e-01]
     [  3.81103879e-01   3.77165970e-01  -2.79344311e-01]
     [  3.89403608e-01   3.05662039e-02   9.24004739e-01]
     [  4.48582217e-01   1.74794018e-04   1.16001176e+00]]
    [  0.2452701    2.79216704   0.36777408   1.02298609  16.60788446]
    [ 0.  0.  0.  0.  0.]
    Random Vector: Independent Random Vector
    Domain: Rectangular Domain < R^3
    Rectangle: [[ 0.69314718         inf]
     [ 0.          1.        ]
     [       -inf         inf]]
    pdf of domain: 0.5
    [[  5.19548316e-01   7.68241112e-01   3.91270986e-01]
     [  1.39697221e-01   1.45666923e-02  -4.77341007e-01]
     [  3.81103879e-01   3.77165970e-01  -2.79344311e-01]
     [  3.89403608e-01   3.05662039e-02   9.24004739e-01]
     [  4.48582217e-01   1.74794018e-04   1.16001176e+00]]
     [ 2.47788783  0.073047    4.95662696  0.16646329  0.09860328]


.. _kle:

Karhunen-Loeve Expansion
------------------------

`Karhunen-Loeve Expansion <http://en.wikipedia.org/wiki/Karhunen%E2%80%93Lo%C3%A8ve_theorem>`_
(KLE) is a way to represent random fields with a discrete set of
random variables. It can be thought of as a discrete representation
of a random field, or a low dimensional representation of a
high-dimensional random vector.
It is implemented via the class
:class:`best.random.KarhunenLoeveExpansion`.

.. class:: best.random.KarhuneLoeveExpansion

    :inherits: :class:`best.random.RandomVector`
    Define a Discrete Karhunen-Loeve Expansion.
    It can also be thought of a as a random vector.

    .. attribute:: PHI

        Get the set of eigenvectors of the covariance matrix.

    .. attribute:: lam

        Get the set of eigenvalues of the covariance matrix.

    .. attribute:: sigma

        Get the signal strength of the model.

    .. method:: __init__(PHI, lam[, mean=None[, sigma=None[, \
                         name='Karhunen-Loeve Expansion']]])

        Initialize the object.

        :param PHI: The eigenvector matrix.
        :type PHI: 2D numpy array
        :param lam: The eigen velues.
        :type lam: 1D numpy array
        :param mean: The mean of the model.
        :type mean: 1D numpy array

        :precondition: ``PHI.shape[1] == lam.shape[1]``.

    .. method:: _eval(theta, hyp)

        Evaluate the expansion at ``theta``.

        :note: Do not use this directly. Use
               :func:`best.random.KarhunenLoeveExpansion.__call__()`
               which is directly inherited from
               :class:`best.maps.Function`.

        :param theta: The weights of the expansion.
        :type theta: 1D array
        :param hyp: Ignored.
        :overloads: :func:`best.maps.Function._eval()`

    .. method:: project(y)

        Project ``y`` to the space of the KLE weights.
        It is essentially the inverse of
        :func:`best.random.KarhunenLoeveExpansion.__call__()`.

        :param y: A sample from the output.
        :type y: 1D numpy array
        :returns: The weights corresponding to ``y``.
        :rtype: 1D numpy array

    .. _rvs(self)

        Return a sample of the random vector.

        :note: Do not use this directly. Use
               :func:`best.random.KarhuneLoeveExpansion.rvs()`
               which is directly inherited from
               :class:`best.random.RandomVector`.

        :returns: A sample of the random vector.
        :rtype: 1D numpy array
        :overloads: :func:`best.random.RandomVector._rvs()`

    .. method:: _pdf(self, y)

        Evaluate the pdf of the random vector at ``y``.

        :note: Do not use this directly. Use
               :func:`best.random.KarhuneLoeveExpansion.pdf()`
               which is directly inherited from
               :class:`best.random.RandomVector`.

        :returns: The pdf at ``y``.
        :rtype: ``float``
        :overloads: :func:`best.random.RandomVector._pdf()`

    .. method:: create_from_covariance_matrix(A[, mean=None[ \
                                              energy=0.95[, \
                                              k_max=None]]])

        Create a :class:`best.random.KarhuneLoeveExpansion` object
        from a covariance matrix ``A``.

        This is a static method.

        :param A: The covariance matrix
        :type A: 2D numpy array
        :param mean: The mean of the model. If ``None`` then all zeros.
        :type mean: 1D numpy array or ``None``
        :param energy: The energy of the field you wish to retain.
        :type energy: ``float``
        :param k_max: The maximum number of eigenvalues to be computed. \
                      If ``None``, then we compute all of them.
        :type k_max: ``int``
        :returns: A KLE based on ``A``.
        :rtype: :class:`best.random.KarhunenLoeveExpansion`.

Let's now look at a simple 1D example::

    import numpy as np
    import matplolib.pyplot as plt
    from best.maps import CovarianceFunctionSE
    from best.random import KarhunenLoeveExpansion
    # Construct a 1D covariance function
    k = CovarianceFunctionSE(1)
    # We are going to discretize the field at:
    x = np.linspace(0, 1, 50)
    # The covariance matrix is:
    A = k(x, x, hyp=0.1)
    kle = KarhunenLoeveExpansion.create_from_covariance_matrix(A)
    # Let's plot 10 samples
    plt.plot(x, kle.rvs(size=10).T)
    plt.show()

You should see something like the following figure:

    .. figure:: images/kle_1d.png
        :align: center

        Samples from a 1D Gaussian random field with zero mean and
        a :ref:`cov-se` using the :ref:`kle`.