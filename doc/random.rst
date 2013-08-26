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

    .. method:: _rvs(self)

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


.. _like:

Likelihood Functions
--------------------

Many algorithms require the concept of a likelihood function. This is
provided via:

.. class:: best.random.LikelihoodFunction

    :inherits: :class:`best.maps.Function`

    The base class of all likelihood functions.

    A likelihood function is a actually a function of the
    hyper-parameters (or simply the parameters) of the model
    and the data. In Bayesian statistics,
    it basicaly models:

    .. math::
        L(x) = \ln p\left(\mathcal{D} |x\right)
        :label: likelihood

    :math:`\mathcal{D}` is the data and it should be set either
    at the constructor or with::

        likelihood.data = data

    The log of the likelihood at :math:`x` for a given
    :math:`\mathcal{D}` (see :eq:`likelihood`) is evaluated by::

        likelihood.__call__(x),

    which is a function that should be implemented by the user.

    .. method:: __init__(num_input[, data=None[, \
                         name='Likelihood Function'[, \
                         log_l_wrapped=None]]])

        Initialize the object.

        :param num_input: The number of inputs (i.e., the number of
                          parameters of the likelihood function).
        :type num_input: ``int``
        :param data: The observed data.
        :param name: A name for the distribution.
        :param log_l_wrapped: A normal function that implements,
                              the likelihood.

    .. attribute:: data

        Set/Get the data. The data can be any object.


A more specific likelihood class that accepts only real data is the
following:


.. class:: best.random.LikelihoodFunctionWithGivenMean

    :inherits: :class:`best.random.LikelihoodFunction`

    This class implements a likelihood function, that requires the evaluation
    of another function within it (.e.g. that of a forward solver.). It is not
    to be used by its own, but rather as a base class for more specific likelihood
    functions.

    Here, we assume that the data are actually a num_data vector and that the
    mean of the likelihood is given by a function (mean_function) with num_input
    variables to num_data variables.


    .. method:: __init__([num_input=None[, data=None[, \
                          mean_function=None[, \
                          name='Likelihood Function with given mean']]]])

        Initializes the object.

        .. warning::

            Either num_input or mean_function must be specified.
            If mean_function is a simple function, then the data are required
            so that we can figure out its output dimension.

        :param num_input: Number of input dimensions. Ignored, if
                          mean_function is a Function class.
        :param data: The observed data. A numpy vector. It must
                          be specified if mean_function is a normal
                          function.
        :param mean_function: A Function or a normal function. If, it is
                              a Function, then mean_function.num_output
                              must be equal to data.shape[0]. If it is
                              a normal function, then it is assumed that
                              it returns a data.shape[0]-dimensional vector.
        :param name: A name for the likelihood function.

    .. method:: _to_string(pad):

        :overloads: :func:`best.maps.Function._to_string()`

    .. attribute:: data

        :overloads: :attr:`best.random.LikelihoodFunction.data`

    .. attribute:: num_data

        Get the number of dimensions of the data.

    .. attribute:: mean_function

        Set/Get the mean function.

.. _like-gauss:

Gaussian Likelihood Function
++++++++++++++++++++++++++++

Here is a class that implements a Gaussian likelihood function:

.. class:: best.random.GaussianLikelihoodFunction

    :inherits: :class:`best.random.LikelihoodFunctionWithGivenMean`

    This class represents a Gaussian likelihood function:

    .. math::
        p\left( \mathcal{D} | x\right) =
        \mathcal{N}\left( \mathcal{D} | f(x), C\right),

    where :math:`C` is the covariance matrix and :math:`f(\cdot)`
    a mean function.

    .. method:: __init__([num_input=None[, data=None[, \
                          mean_function=None[, cov=None[, \
                          name='Gaussian Likelihood Function']]]]])

        Initialize the object.

            :param num_input: The number of inputs. Optional, if
                              mean_function is a proper
                              :class:`best.maps.Function`.
            :param data: The observed data. A vector. Optional,
                         if ``mean_function`` is a proper
                         :class:`best.maps.Function`.
                         It can be set later.
            :param mean_function: The mean function. See the super class
                                  for the description.
            :param cov: The covariance matrix. It can either be
                        a positive definite matrix, or a number.
            :param name: A name for the likelihood function.

            :precondition: You must provide eithe the ``data`` or a
                           proper ``mean_function``.

    .. method:: __call__(x)

        :overloads: :func:`best.maps.Function.__call__()`

    .. attribute:: cov

        Set/Get the covariance matrix.

.. _like-student:

Student-t Likelihood Function
+++++++++++++++++++++++++++++

Here is a class that implements a Student-t likelihood function:

.. class:: best.random.StudentTLikelihoodFunction

    :inherits: :class:`best.random.Gaussian`

    This class represents a Student-t likelihood function.

    .. method:: __init__(nu, [num_input=None[, data=None[, \
                          mean_function=None[, cov=None[, \
                          name='Gaussian Likelihood Function']]]]])

        Initialize the object.

            :param nu: The degrees of freedom of the distribution.
            :param num_input: The number of inputs. Optional, if
                              mean_function is a proper
                              :class:`best.maps.Function`.
            :param data: The observed data. A vector. Optional,
                         if ``mean_function`` is a proper
                         :class:`best.maps.Function`.
                         It can be set later.
            :param mean_function: The mean function. See the super class
                                  for the description.
            :param cov: The covariance matrix. It can either be
                        a positive definite matrix, or a number.
            :param name: A name for the likelihood function.

            :precondition: You must provide eithe the ``data`` or a
                           proper ``mean_function``.

    .. method:: __call__(x)

        :overloads: :func:`best.random.GaussianLikelihoodFunction.__call__()`


.. _dist:

Distributions
-------------

A distribution is a concept that combines a likehood function with
a random variable. Most probably we will replace it in the future
with the :class:`best.random.RandomVector` class that combines all
these concepts.

.. class:: best.random.Distributions

    :inherits: :class:`best.random.LikelihoodFunction`

    The base class of all distributions.

    .. method:: __init__(num_input[, name='Distribution')

        Initialize the object.

        :param num_input: The number of input dimensions.
        :type num_input: ``int``
        :param name: A name for the distribution.
        :type name: ``str``

    .. method:: sample([x=None])

        Sample the distribution.

        :param x: If it is specified then x should be overriden.
                  If it is not specified, then the sample is allocated and
                  returned.


Many distributions together can be represented by:

.. class:: best.random.JointDistribution

    :inherits: :class:`best.random.Distribution`

    A class that represents a collection of random variables.

    .. method:: __init__(dist[, name='Joint Distribution'])

        Initialize the object.

        :param dist: A list of distributions to be joined.
        :type dist: :class:`best.random.Distribution`

    .. method:: __call__(x)

        :overloads: :func:`best.maps.Function.__call__()`

    .. method:: sample([x=None])

        :overloads: :func:`best.random.Distribution.sample()`

    .. attribute:: dist

        Get the underlying distributions


A conditional distribution can be represented by:


.. class:: best.random.ConditionalDistribution

    :inherits: :class:`best.random.Distribution`

    The base class for conditional distributions.

    .. method:: __init__(num_input, num_cond[, \
                         name='Conditional Distribution')

        Initialize the object.

        :param num_input: The number of inputs.
        :param num_cond: The number of conditioning variables.

    .. method:: __call__(z, x)

        :redefines: :func:`best.maps.Fucntion.__call__()`

        Evaluate the log of the probability at ``z`` given ``x``.

        :throws: :class:`NotImplementedError`

    .. method:: sample(x[, z=None])

        :redefines: :func:`best.random.Distribution.sample()`

        Sample ``z`` given ``x``.

        :throws: :class:`NotImplementedError`

    .. attribute:: num_cond

        Get the number of conditioning variables.


The following class represents the product of a distribution with
a conditional distribution:

    .. math::

        p(z | x)p(x)

Here it is:

.. class:: best.random.ProductDistribution

    :inherits: :class:`best.random.Distribution`

    It corresponds to the product :math:`p(z | x)p(x)`.
    :math:`p(x)` corresponds to ``px`` and :math:`p(z|x)` to ``pzcx``.
    The input is assumed to be the vector ``[x, z]``.

    .. method:: __init__(pzcx, px[, name='Product Distribution'])

        Initialize the object.

        :param pzcx: The distribution :math:`p(z | x)`.
        :type pzcx: :class:`best.random.ConditionalDistribution`
        :param px: The distribution :math:`p(x)`.
        :type px: :class:`best.random.Distribution`

    .. method:: __call__(x)

        :overloads: :func:`best.maps.Function.__call__()`

    .. method:: sample()

        :overloads: :func:`best.maps.Distributions.sample()`

    .. attribute:: px

        Get the distribution corresponding to :math:`p(x)`.

    .. attribute:: pzcx

        Get the distribution corresponding to :math:`p(z | x)`


.. _dist-sampling:

Sampling Distributions
----------------------

We provide several distributions that you can use right away to
construct new ones.


.. _dist-uniform:

Uniform Distribution
++++++++++++++++++++

.. class:: best.random.UniformDistribution

    :inherits: :class:`best.random.Distribution`

    The uniform distribution on a square domain.

    .. method:: __init__(num_input[, domain=None[, \
                         name='Uniform Distribution']])

        Initialize the object.

        :param num_input: The number of dimensions.
        :param domain:  The domain of the random variables. Must be a (k x 2)
                        matrix. If not specified, then a unit hyper-cube is
                        used.
        :param name: A name for this distribution.

    .. method:: __call__(x)

        :overloads: :func:`best.maps.Function.__call__()`

        Evaluates the logarithm of:

            .. math:: p(x) = \frac{1}{|D|},

        where :math:`|D|` is the measure of the domain.

    .. method:: sample([x=None])

        :overloads: :func:`best.random.Distribution.sample()`

    .. attribute:: domain

        Get/Set the domain.


.. _dist-norm:

Normal Distribution
+++++++++++++++++++

.. class:: best.random.NormalDistribution

    :inherits: :class:`best.random.Distribution`

    .. method:: __init__(num_input[, mu=None[, cov=None[, \
                         name='Normal Distribution')

        Initialize the object.

        :param num_input: The dimension of the random variables.
        :param mu: The mean. Zero if not specified.
        :param cov: The covariance matrix. Unit matrix if not specified.
        :param name: A name for the distribution.

    .. method:: __call__(x)

        :overloads: :func:`best.maps.Function.__call__()`

    .. method:: sample([x=None])

        :overloads: :func:`best.random.Distribution.sample()`


.. _dist-student:

Student-t Distribution
++++++++++++++++++++++

.. class:: best.random.StudentTDistribution

    :inherits: :class:`best.random.NormalDistribution`

    .. method:: __init__(num_input, nu[, mu=None[, cov=None[, \
                         name='Normal Distribution')

        Initialize the object.

        :param num_input: The dimension of the random variables.
        :param nu: The degrees of freedom.
        :param mu: The mean. Zero if not specified.
        :param cov: The covariance matrix. Unit matrix if not specified.
        :param name: A name for the distribution.

    .. method::__call__(x)

        :overloads: :func:`best.random.NormalDistribution.__call__()`

    .. sample([x=None])

        :overloads: :func:`best.random.NormalDistribution.sample()`


.. _mixture:

Mixture of Distributions
------------------------

Here is a class that represents a mixture of distributions:

.. class:: best.random.MixtureOfDistributions

    :inherits: :class:`best.random.Distribution`

    A class representing a mixture of distributions:

    .. math::

        p(x) = \sum_i c_i p_i(x),

    where :math:`\sum_i c_i = 1` and :math:`p_i(x)` are distributions.

    .. method:: __init__(weights, components[, \
                         name='Mixture of Distributions')

        Initialize the object.

        :param weights: The weight of each component.
        :param components: A list of the components.
        :type components: :class:`best.random.Distribution`

    .. method:: __call__(x)

        :overloads: :func:`best.maps.Function.__call__()`

    .. method:: sample([x=None])

        :overloads: :func:`best.random.Distribution.sample()`

    .. attribute:: weights

        Get the weights of each component.

    .. attribute:: components

        Get the components (distributions).

    .. attribute:: num_components

        Get the number of components.


.. _post:

Posterior Distribution
----------------------

A class representing a posterior distribution. It requires a likelihood
function and a prior. This is design to be used with :ref:`smc`.
This is why it is a little bit strange. Most probably, you won't
have to overload it (or even understand how it works) unless you
are doing something very special.

.. class:: best.random.PosteriorDistribution

    :inherits: :class:`best.random.LikelihoodFunction`

    A class representing a posterior distribution.

    **The likelihood function:**
    The class requires a likelihood object which can be any class implementing:

        + ``likelihood.__call__(x)``:   Giving the log likelihood at a particular x.
          Notice that x here is the parameter of the likelihood not the data.
          It is the responsibility of the user to make sure that the likelihood function,
          correctly captures the dependence on the data.

    **The prior distribution:**
    The prior distribution is any object which implements:

        + ``prior.__call__(x)``: Giving the log of the prior.

    Overall, this class represents the following function:

        .. math::
            p(x | y, \gamma) \propto p(y | x)^\gamma p(x).

    Again, I mention that specifying :math:`y` is the responsibility of the user.
    It is not directly needed in this class. All we use is
    :math:`p(y | x)` as a
    function :math:`x` only, :math:`y` being fixed and implied.
    The parameter gamma plays the role of a regularizing parameter.
    The default value is 1. We have explicitely included it, because the main
    purpose of this class is to be used within the :ref:`smc`
    framework.

    .. method:: __init__([likelihood=None[, prior=None[, gamma=1[, \
                          name='Posterior Distribution']]]])

        Initialize the object.

        :param likelihood: The likelihood function.
        :param prior: The prior distribution.
        :param gamma: The regularizing parameter.
        :type gamma: ``float``
        :param name: A name for the distribution.

    .. method:: __call__(x[, report_all=False])

        :overloads: :func:`best.maps.Function.__call__()`

        Evaluate the log of the posterior at ``x``.

        :param x: The point of evalutation.
        :param report_all:      If set to True, then it returns
                                a dictionary of all the values used
                                to compose the log of the posterior (see
                                below for details). Otherwise, it simply
                                returns the log of the posterior.

        The function returns a dictionary r that contains:
            + ``r['log_p']``:       The log of the pdf at x.
            + ``r['log_like']``:    The log of the likelihood at x.
            + ``r['log_prior']``:   The log of the prior at x.
            + ``r['gamma']``:       The current gamma.

    .. method:: _to_string(pad)

        :overloads: :func:`best.maps.Function._to_string()`

    .. attribute:: likelihood

        Get/Set the likelihood function.

    .. attribute:: prior

        Get/Set the prior.

    .. attribute:: gamma

        Get/Set gamma


.. _mcmc:

Markov Chain Monte Carlo
------------------------

We start by listing all the classes that take part in the formulation
of the MCMC sampling algorithm. The purpose here, is to highlight the
inheritance structure. We will give an example at the end that puts
everything together.

The base class of any Markov Chain should be a:

.. class:: best.random.MarkovChain

    :inherits: :class:`best.Object`

    The base class for a Markov Chain. Any Markov Chain should inherit
    it.

    .. method:: __init__([name='Markov Chain'])

        Initialize the object.

    .. method:: __call__(x_p, x_n)

        Evaluate the logarithm of the pdf of the chain.

        Usually this would be written as:

            .. math:: \ln p(x_n | x_p),

        but we are using the programming convention that whatever is
        given comes first.

        :param x_p: The state on which we are conditioning.
        :param x_n: The new state.
        :trhows: :class:`NotImplementedError`

    .. method:: sample(x_p, x_n)

        Sample from the Markov Chain and write the result on ``x_n``.

        :param x_p: The state on which we are conditioning.
        :param x_n: The new state. To be overwritten.
        :throws: :class:`NotImplementedError`


.. _mcmc-proposal:

Proposal Distributions
++++++++++++++++++++++

MCMC requires a proposal distribution which is, of course, a Markov
Chain:

.. class:: best.random.ProposalDistribution

    :inherits: :class:`best.random.MarkovChain`

    The base class for the proposals used in MCMC.

    .. method:: __init__([dt=1e-3[, name='Proposal Distribution']])

        Initialize the object.

        :param dt: The step size of the proposal. We put this here
                   because many commonly met proposals do have a step
                   size.
        :type dt: ``float``

    .. attribute:: dt

        Get/set the step size of the proposal.


.. _mcmc-random-walk:

Random Walk Proposal
++++++++++++++++++++

A very common proposal distribution is the **Random Walk** proposal:

.. class:: best.random.RandomWalkProposal

    A random walk proposal distribution.

    The chain is defined by:

        .. math:: p(x_n | x_p, \delta t)
                  = \mathcal{N}\left(x_n | x_p, \delta t^2\right).

    .. method:: __init__([dt=1e-3[, name='Random Walk Proposal']])

        Initialize the object.

    .. method:: __call__(x_p, x_n)

        :overloads: :func:`best.random.MarkovChain.__call__()`

    .. method:: __sample__(x_p, x_n)

        :overloads: :func:`best.random.MarkovChain.sample()`

.. _mcmc-langevin:

Langevin Proposal
+++++++++++++++++

The Langevin proposal is implemented via:

.. class:: best.random.LangevinProposal

    :inherits: :class:`best.random.Proposal`

    A Langevin proposal that leads to a Metropolized-Adjusted Langevin
    Algorithm (MALA).

    See the code for further details.


.. _mcmc-class:

The MCMC class
++++++++++++++

Now we are in a position to discuss the implementation of the MCMC
algorithm in :mod:`best`. It is achieved via the class
:class:`best.random.MarkovChainMonteCarlo`:

.. class:: best.random.MarkovChainMonteCarlo

    A general MCMC sampler.

    **The state of MCMC.**
    We assume that the state ``x`` of MCMC is a class that implements
    the following methods:

        + ``x.copy()``: Create a copy of ``x`` and returns a reference to it.

    **The proposal of MCMC.**
    The proposal of MCMC should implement the following methods:

        + ``proposal.__call__(x_p, x_n)``: Evaluate the log of the pdf of
                                           :math:`p(x_n | x_p)` and return the result.
                                           For reasons of computational efficiency,
                                           we had to make the return value of this
                                           function a little bit more complicated than
                                           is necessary in MCMC. We assume that it
                                           returns an dictionary obj that has at least one
                                           field:

                                               + ``obj['log_p']``: The logarithm of the pdf at ``x``.

                                           This object corresponding to the current state
                                           is always stored.
                                           To understand, why something this awkward is
                                           essential, please see the code of the
                                           :class:`best.random.SequentialMonteCarlo` class.

        + ``proposal.sample(x_p, x_n)``:   Sample :math:`p(x_n | x_p)` and write the
                                           result on ``x_n``.

    **The target distribution.**
    The target distribution should implement:

        + ``target.__call__(x)``: Evaluate the log of the target pdf up
                                  to a normalizing constant.

    .. method:: __init__([target=None[, \
                          proposal=RandomWalkProposal()[,
                          store_samples=False[, verbose=False[, \
                          output_frequency=1]]]]])

        Initialize the object.

        :param target: The target distribution.
        :param proposal: The proposal distribution.
        :param store_samples: If set to ``True``, then all samples are stored
                           (copied) and are accessible via self.samples.
        :param verbose: The verbosity flag. If set to True, then sth
                        is printed at each MCMC step.
        :param output_frequency: If verbose is ``True``, then this specifies how often
                                 something is printed.

    .. method:: initialize(x[, eval_state=None])

        Initialize the chain.

        Initializes the chain at ``x``. It is essential that the chain has been
        properly initialized!

    .. method:: reinitialize()

        Re-initialize the chain.

    .. method:: perform_single_mcmc_step(self)

        Performs a single MCMC step.

        The current state of the chain is altered at the end (or not).

    .. method:: sample([x=None[, eval_state=None[, \
                        return_eval_state=False[, steps=1]]]])

        Sample the chain.

        :param x: The initial state. If not specified, then
                  we assume that it has already been set.
        :param steps: The number of MCMC steps to be performed.
        :param return_eval_state: Return the evaluated state at the end of
                                  the run.
        :returns: A reference to the current state of the chain.

    .. method:: copy()

        Return a copy of this object.

    .. attribute:: target

        Set/Get the target distribution.

        Every time the target changes, the chain must be initialized again.
        If the current state is already present, then this method automatically
        reinitializes the chain.

    .. attribute:: proposal

        Set/Get the proposal

    .. attribute:: current_state

        Get the current state of MCMC.

    .. attribute:: proposed_state

        Get the proposed state of MCMC.

    .. attribute:: num_samples

        Get the number of samples taken so far.

    .. attribute:: num_accepted

        Get the number of accepted samples so far.

    .. attribute:: acceptance_rate

        Get the acceptance rate.

    .. attribute:: initialized

        Check if the chain has been initialized.

    .. attribute:: store_samples

        Check if the samples are being stored.

    .. attribute:: samples

        Get the stored samples.

    .. attribute:: verbose

        Get/Set the verbosity flag.

    .. attribute:: output_frequency

        Get/Set the output_frequency.


.. _mcmc-example:

Simple MCMC Example
-------------------

Now that we have introduced :class:`best.random.MarkovChainMonteCarlo`,
let's look at a very simple example that can be found in
:file:`best/demo/test_mcmc.py`::

    if __name__ == '__main__':
        import fix_path

    import numpy as np
    import math
    from best.random import *
    import matplotlib.pyplot as plt


    class SampleTarget(LikelihoodFunction):
        """A sample target distribution."""

        def __init__(self):
            super(SampleTarget, self).__init__(1)

        def __call__(self, x):
            k = 3.
            t = 2.
            if x[0] < 0.:
                return -1e99
            else:
                return (k - 1.) * math.log(x[0]) - x[0] / t


    if __name__ == '__main__':
        target = SampleTarget()
        x_init = np.ones(1)
        proposal = RandomWalkProposal(dt=5.)
        mcmc = MarkovChainMonteCarlo(target=target, proposal=proposal,
                                     store_samples=True,
                                     verbose=True,
                                     output_frequency=1000)
        mcmc.initialize(x_init)
        mcmc.sample(steps=100000)
        samples = [mcmc.samples[i][0] for i in range(len(mcmc.samples))]
        plt.hist(samples, 100)
        plt.show()

This should plot the following figure:

    .. figure:: images/mcmc_1d.png
        :align: center

        The histogram of the samples gathered by MCMC.


.. _smc:

Sequential Monte Carlo
----------------------

Sequential Monte Carlo (SMC) is a way to sample multi-modal probability
distributions by constructing a sequence of distributions that converge
to the target distribution in a smooth manner and propagating through
them an ensemble of particles.

In :mod:`best` SMC is implemented via the
:class:`best.random.SequentialMonteCarlo` which:

    + Can work with arbitrary underlying MCMC samplers.
    + Can automatically detect an optimal sequence of distributions.
    + Can be run in parallel.

The mathematical details can be found in our paper on inverse problems
which is currently under review.

.. class:: best.random.SequentialMonteCarlo

    For the moment do the following to get the complete documentation:

    >> from best.random import SequentialMonteCarlo
    >> help(SequentialMonteCarlo)

    I will add the complete documentation in short time.


.. _smc-example:

Simple Sequential Monte Carlo Example
-------------------------------------

We are going to use :ref:`smc` to sample from a mixture of Gaussians::

    if __name__ == '__main__':
        import fix_path


    import numpy as np
    import math
    from best.random import *
    import matplotlib.pyplot as plt


    if __name__ == '__main__':
        # Number of inputs
        num_input = 1
        # Construct the likelihood function
        # Number of components
        num_comp = 4
        # Degrees of freedom of the Inverse Wishart distribution
        # from which we draw the covariance matrix
        n_d = 10
        # Randomly pick each component
        components = []
        for i in range(num_comp):
            mu = 5. * np.random.randn(num_input)
            X = np.random.randn(n_d, num_input)
            A = np.dot(X.T, X)
            C = np.linalg.inv(A)
            components.append(NormalDistribution(num_input, mu, C))
        # Randomly pick weights for the components
        #w = np.random.rand(num_comp)
        w = np.ones(num_comp) / num_comp
        # Construct the likelihood
        likelihood = MixtureOfDistributions(w, components)
        # Let's just take a look at this distribution
        print 'weights:, ', likelihood.weights
        print 'components:'
        for c in likelihood.components:
            print 'mu: ', c.mu
            print 'cov: ', c.cov
        x = np.linspace(-10., 10., 100.)
        # The prior is just going to be a normal distribution with
        # zero mean and very big variance
        prior = NormalDistribution(num_input, cov=2.)
        # Construct the SMC object
        smc = SequentialMonteCarlo(prior=prior, likelihood=likelihood,
                                   verbose=True, num_particles=1000,
                                   num_mcmc=10,
                                   proposal=RandomWalkProposal(dt=2.),
                                   store_intermediate_samples=True)
        r, w = smc.sample()
        step = 0
        for s in smc.intermediate_samples:
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.hist(s['r'], bins=20, weights=s['w'], normed=True)
            ax1.set_xlim([-5., 5.])
            ax1.set_title('gamma = %1.4f' % s['gamma'])
            ax1.set_xlabel('x')
            ax1.set_ylabel('normalized histogram')
            ax2 = fig.add_subplot(1, 2, 2)
            smc.mcmc_sampler.target.gamma = s['gamma']
            log_post = np.array([smc.mcmc_sampler.target(np.array([t])) for t in x])
            ax2.plot(x, np.exp(np.exp(log_post)))
            ax2.set_title('gamma = %1.4f' % s['gamma'])
            ax2.set_xlabel('x')
            ax2.set_ylabel('pdf')
            plt.savefig('smc_step=%d.png' % step)
            step += 1

The code will produce a sequence of ``*.png`` files showing the evolution
of the algorithm:

    .. figure:: images/smc_step=0.png
        :align: center

    .. figure:: images/smc_step=1.png
        :align: center

    .. figure:: images/smc_step=2.png
        :align: center

    .. figure:: images/smc_step=3.png
        :align: center

    .. figure:: images/smc_step=4.png
        :align: center

    .. figure:: images/smc_step=5.png
        :align: center