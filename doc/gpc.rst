Generalized Polynomial Chaos
============================

.. module:: best.gpc
    :synopsis: Classes and methods for the construction of Generalized
               Polynomial Chaos (gPC) with arbitrary probability
               distributions.

The module :mod:`best.gpc` implements functionality related to the
construction and manipulation of orthogonal polynomials with respect
to arbitrary probability distributions. Mathematically, let
:math:`f(\cdot)` be a probability distribution. Then it allows the
construction of polynomials :math:`\phi_n(\cdot)` such that:

.. math::
    \int \phi_n(\mathbf{x})\phi_m(\mathbf{x})f(\mathbf{x})d\mathbf{x}
    = \delta_{nm}.

It is largly based on the legacy Fortran code ORTHPOL.


1D Orthogonal Polynomials
-------------------------

We start with the construction of polynomials in one dimension. This
can be achieved via the class :class:`best.gpc.OrthogonalPolynomial`
which is described below.

.. class:: best.gpc.OrthogonalPolynomial

    A class representing a 1D orthogonal polynomial with respect to a \
    particular probability density.

    The polynomials are described internally via the recursive relations:

        .. math::
            \phi_n(x) = \left((x - \alpha_{n-1})\phi_{n-1}(x)
            - \beta_{n-1}\phi_{n-2}(x)\right) / \gamma_{n},

    with

        .. math::
            \phi_0(x) = 1 / \gamma_0

    and

        .. math::
            \phi_1(x) = (x - \alpha_0)\phi_0(x) / \gamma_1.

    They are always properly normalized. Keep in mind that it inherits
    from :class:`best.maps.Function`, so it is a multi-output function.
    The number of outputs is essentially equal to the number of
    polynomials represented by the class.

    .. method:: __init__(degree[, left=-1[, right=1[, wf=lambda(x): 1.[, \
                 ncap=50[, quad=QuadratureRule[, \
                 name='Orthogonal Polynomial']]]]]])

        Initialize the object.

        The polynomial is constructed on an interval :math:`[a, b]`,
        where :math:`-\infty \le a < b \le +\infty`. The construction
        of the polynomial uses the Lanczos procedure
        (see :func:`best.gpc.lancz()`) which relies on a quadrature
        rule. The default quadrature rule is the n-point Fejer rule
        (see :func:`best.gpc.fejer()`). These default can, of course,
        be bypassed by overloading this class.

        :param degree: The degree of the polynomial.
        :type degree: int
        :param left: The left side of the interval over which the \
                     polynomial is defined. :math:`-\infty` may be
                     specified by ``-float('inf')``.
        :type left: float
        :param right: The right side of the interval over which the \
                      polynomial is defined. :math:`\infty` may be
                      specified by ``float('inf')``.
        :type right: float
        :param f: The probability density serving as weight.
        :type wf: A real function of one variable.
        :param ncap: The number of quadrature points to be used.
        :type ncap: int
        :param quad: A quadrature rule. See the description below.
        :type quad: :class:`best.gpc.QuadratureRule`
        :param name: A name for the polynomials.
        :type name: str

    .. attribute:: degree

        The maximum degree of the polynomials.
        The number of polynomials is ``degree + 1``.

    .. attribute:: alpha

        The :math:`\alpha_n` coefficients of the recursive relation.

    .. attribute:: beta

        The :math:`\beta_n` coefficients of the recursive relation.

    .. attribute:: gamma

        The :math:`\gamma_n` coefficients of the recursive relation.

    .. method:: __call__(x)

        Evaluate the polynomials at a particular point.

        :param x: The evaluation point or an array of many evaluation points.
        :type x: float or 1D numpy array
        :returns: The value of all the polynomials at all points in ``x``.
        :rtype: 2D numpy array

    .. method:: d(x)

        Evaluate the derivative of the polynomials at a particular point.

        :param x: The evaluation point or an array of many evaluation points.
        :type x: float or 1D numpy array
        :returns: The value of the derivatives of all the polynomials at \
                  all points in ``x``.
        :rtype: 2D numpy array

    .. is_normalized()

        Return ``True`` if the polynomials have unit norm and ``False`` \
        otherwise.

        :Note: If you use the default constructor the polynomials are \
               automatically normalized.

    .. normalize()

        Normalize the polynomials so that they have unit norm.

Constructing Polynomials
------------------------

We now show how to construct several of the standar orthogonal
polynomials used in the literature. For convenience, assume that in all
examples we have imported the ``matplotlib.pyplot`` of the
`matplotlib library <http://matplotlib.org/>`_ by::

    import matplotlib.pyplot as plt

Hermite polynomials
+++++++++++++++++++

The Hermite polynomials are defined on :math:`(-\infty, \infty)` are
orthogonal with respect to the probability density:

    .. math:: f(x) = \frac{1}{\sqrt{2\pi}}e^{-x^2/2}.

Here is how you can construct them up to degree 10::

    from best.gpc import OrthogonalPolynomial
    infty = float('inf')    # A number representing infinty.
    degree = 10             # The degree of the polynomials
    wf = lambda(x): 1. / math.sqrt(2. * math.pi) * np.exp(-x ** 2 / 2.)
    p = OrthogonalPolynomial(degree, left=-infty, right=infty, wf=wf)

Notice the definition of the probability density function. This could
be a regular function or a :class:`best.maps.Function`. Here we have
opted for the much quicker lambda structure.

You may look at the coefficients of the recursive formula by::

    print str(p)

which should produce the following text::

    Orthogonal Polynomial:R^1 --> R^11
     alpha: [  8.24721933e-17   3.00634774e-16  -1.87973171e-16   3.50005961e-16
      -5.28859926e-16   4.05750024e-16  -6.69888614e-16  -8.13357045e-16
       8.06209016e-16  -2.21298838e-15   7.45252594e-16]
     beta: [ 1.00000012  1.00000151  1.41419988  1.73184818  2.0000218   2.24118187
       2.45731915  2.59905334  2.71187277  3.16796779  3.68960306]
     gamma: [ 1.00000012  1.00000151  1.41419988  1.73184818  2.0000218   2.24118187
       2.45731915  2.59905334  2.71187277  3.16796779  3.68960306]
     normalized: True

You can see the polynomials by::

    x = np.linspace(-2., 2., 100)
    plt.plot(x, p(x))
    plt.show()

which should produce the following figure:

    .. figure:: images/hermite.png
        :align: center

        The Hermite polynomials up to degree 10.

Similarly you may visualize their derivatives by::

    plt.plot(x, p.d(x))
    plt.show()

which should produce the following figure:

    .. figure:: images/hermite_der.png
        :align: center

        The derivative of the Hermite polynomials up to degree 10.


Laguerre polynomials
++++++++++++++++++++
The Laguerre polynomials are defined on :math:`(0, \infty)` are
orthogonal with respect to the probability density:

    .. math:: f(x) = e^{-x}.

Up to degree 10, they may be constructed by::

    from best.gpc import OrthogonalPolynomial
    infty = float('inf')    # A number representing infinty.
    degree = 10             # The degree of the polynomials
    wf = lambda(x): np.exp(-x)
    p = OrthogonalPolynomial(degree, left=0, right=infty, wf=wf)

Here is how they look:

    .. figure:: images/laguerre.png
        :align: center

        The Laguerre polynomials up to degree 10.


Exploiting :mod:`scipy.stats`
++++++++++++++++++++++++++++++++++++++++++

It is also possible to use functionality from scipy to define the
probability density. For example, you may construct the Laguerre
polynomials by::

    import scipy.stats
    # Define the random variable
    rv = scipy.stats.expon()
    p = OrthogonalPolynomial(degree, left=0, right=infty, wf=pdf)

This is a nice trick, because you can immediately construct any
orthogonal polynomial you wish making use of the probability
distributions that can be found in
`scipy.stats <http://docs.scipy.org/doc/scipy/reference/stats.html>`_.
All you need to do is:

    1. Construct a random variable ``rv``.
    2. USe ``rv.pdf`` as the weight function when constructing the
       :class:`best.gpc.OrthogonalPolynomial`.

Here are for example orthogonal polynomials with respect to the Beta
distribution:

    .. math::

        f(x) = \frac{\Gamma(a + b)}{\Gamma(a)\Gamma(b)} x^{a - 1}
               (1 - x)^{b - 1},

with :math:`a, b>0` and :math:`x \in (0, 1)`::

        import scipy.stats
        a = 0.3
        b = 0.8
        rv = scipy.stats.beta(a, b)
        p = best.gpc.OrthogonalPolynomial(6, left=0, right=1, wf=rv.pdf)

Here are the first six:

    .. figure:: images/beta.png
        :align: center

        The first six orthogonal polynomials with respect to the Beta \
        distribution with :math:`a = 0.3, b = 0.8`.