.. _rvm:

Relevance Vector Machine
========================

.. module:: best.rvm
    :synopsis: Defines the class that trains the Relevance Vector \
               Machine (RVM).

Relevance Vector Machine (RVM) trains a :ref:`glm`
yielding sparse representation (i.e., many of the basis functions are
not used at the end). The implementation in BEST is the
Multi-dimensional Relevance Vector Machine (MRVM) as described in
our `paper <http://epubs.siam.org/doi/pdf/10.1137/120861345>`_.
It uses the :ref:`linalg-gsvd` to train the
model, which is considerably more stable than relying to Cholesky
decompositions. This is achieved via the
:class:`best.linalg.GeneralizedSVD` class.

As already said, we start with a :ref:`glm` (see :eq:`glm`):

    .. math::
        \mathbf{f}(\mathbf{x}; \mathbf{W}) =
        \boldsymbol{\phi}(\mathbf{x})^T\mathbf{W},
        :label: glm

where
    .. math::
        \boldsymbol{\phi}(\mathbf{x}) =
        \left(\phi_1(\mathbf{x}), \dots, \phi_m(\mathbf{x})\right)
        :label: basis

forms a :ref:`map-basis`. The :ref:`rvm` trains the model based on a set
of :math:`n` (noisy) observations of a :math:`q` dimensional process:

    .. math::
        \mathcal{D} = \left\{\left(\mathbf{x}^{(i)},
                              \mathbf{y}^{(i)}\right) \right\}_{i=1}^n.
        :label: data
It assigns a Gaussian noise with precision (inverse noise) :math:`\beta`
to the observations (this defines the likelihood) and the following
prior on the weight matrix :math:`\mathbf{W}`:

    .. math::
        p(\mathbf{W}|\boldsymbol{\alpha}) =
        \prod_{j=1}^qp(\mathbf{W}_j|\boldsymbol{\alpha}),
        :label: joint-prior

    .. math::
        p(\mathbf{W}_j | \boldsymbol{\alpha}) =
        \prod_{i=1}^mp(W_{ij} | \alpha_i),
        :label: joint-prior-2

    .. math::
        p(W_{ij} | \alpha_i) \propto
        \exp\left\{-\alpha_{i}W_{ij}^2\right\},
        :label: rvm-prior

where :math:`\boldsymbol{\alpha}` is a set of :math:`m` hyper-parameters,
one for each basis function.
The characteristic of :eq:`rvm-prior` is that as if
:math:`\alpha_i = \infty`, then the basis function
:math:`\phi_i(\cdot)` can be removed for the model.
The parameters :math:`(\boldsymbol{\alpha}, \beta)` are found by
maximizing the evidence of the data :math:`\mathcal{D}`.

The model is realized via the class
:class:`best.rvm.RelevanceVectorMachine` which is described below:

.. class:: best.rvm.RelevanceVectorMachine

    The Relevance Vector Machine Class.

    .. method:: __init__()

        Construct the object. It does nothing.

    .. method:: set_data(PHI, Y)

        Sets the data :math:`\mathcal{D}` to the model.

        ``PHI`` is the design matrix
        :math:`\boldsymbol{\Phi}\in\mathbb{R}^{n\times m}`, where:

            .. math::
                \Phi_{ij} = \phi_j\left(\mathbf{x}^{(i)} \right),

        and ``Y`` is the data matrix
        :math:`\mathbf{Y}\in\mathbb{R}^{n\times q}` in which
        :math:`Y_{ij}` is the :math:`j`-th dimension of the output
        of the :math:`i`-th observed input point.

        :param PHI: The design matrix.
        :type PHI: 2D numpy array
        :param Y: The data matrix.
        :type Y: 2D numpy array

    .. method:: initialize([beta=None[, relevant=None[, alpha=None]]])

        Initialize the algorithm.

        :param beta: The initial beta. If ``None``, then we use the \
                     inverse of the observed variance of the data.
        :type beta: float
        :param relevant: A list of the basis function with which we \
                         wish to start the algorithm. For example, \
                         if ``relevant = [2, 5, 1]``, then basis \
                         functions 2, 5 and 1 are in the model. \
                         The rest have a :math:`\alpha` that is \
                         equal to :math:`\infty`. If ``None``, then \
                         the algorithm will start with a single \
                         basis function, the one whose inclusion \
                         maximizes the evidence.
        :type relevant: array of int
        :param alpha: The values of the :math:`\alpha`'s of the \
                      initial relevant basis functions specified by \
                      ``relevant``.
        :type alpha: 1D numpy array.

    .. method:: train([max_it=10000[, tol=1e-6[, verbose=False]]])

        Train the model.

        :param max_it: The maximum number of iterations of the algorithm.
        :type max_it: int
        :param tol: The convergence tolerance. Convergence is monitored \
                    by looking at the change between consecutive \
                    :math:`\alpha`'s.
        :type tol: float
        :param verbose: Print something or not. The default is ``False``.
        :type verbose: bool

    .. method:: get_generalized_linear_model(basis)

        Construct a :ref:`glm` from the result of the RVM algorithm.

        :param basis: The basis you used to construct the design matrix \
                      ``PHI``.
        :type basis: :class:`best.maps.Function`.

    .. method:: __str__()

        Return a string representation of the object.


A Simple 1D Example
-------------------

Here we demonstrate how the model can be used with a simple 1D example.
We first start with a basis based on Squared Exponential Covariance
(see :ref:`cov-se`) constructed as described in :ref:`cov-basis`::

    import numpy as np
    import matplotlib.pyplot as plt
    import best
    # Number of observations
    num_obs = 100
    # The noise we will add to the data (std)
    noise = 0.1
    # Draw the observed input points randomly
    X = np.random.randn(num_obs)
    # Draw the observations
    Y = np.sin(X) / (X + 1e-6) + noise * np.random.randn(*X.shape)
    # The covariance function
    k = best.maps.CovarianceFunctionSE(1)
    # Construct the basis
    phi = k.to_basis(hyp=2.)
    # Construct the design matrix
    PHI = phi(X)
    # Use RVM on the data
    rvm = best.rvm.RelevanceVectorMachine()
    rvm.set_data(PHI, Y)
    rvm.initialize()
    rvm.train()
    print str(rvm)

This will result in an output similar to::

    Relevant Vector Machine
    Y shape: (100, 1)
    PHI shape: (100, 100)
    Relevant: [ 0 98 59 16 58 65  2 57 68 84 36 93 55 83  3 45]
    Alpha: [  1.75921502   0.04998139   4.35007167   1.87751651   1.12641185
              0.10809376   0.72398214  19.07217688   0.23016274   0.02142343
              0.01976957   2.5164594    1.55757032   0.05801807   0.06522873
              0.61174863]
    Beta: 209.805579349

Now, you may get a :ref:`glm` from the model and plot its mean and
predictive variance::

    f = rvm.get_generalized_lineal_model(phi)
    plt.plot(X, Y, '+')
    x = np.linspace(-10, 10, 100)
    fx = f(x)
    plt.plot(x, fx, 'b', linewidth=2)
    plt.plot(x, np.sin(x) / (x + 1e-6), 'r', linewidth=2)
    # Plot +- 2 standard deviations
    s2 = 2. * np.sqrt(f.get_predictive_variance(x))
    plt.plot(x, fx + s2, 'g')
    plt.plot(x, fx - s2, 'g')
    plt.plot(X[rvm.relevant], Y[rvm.relevant], 'om')
    plt.show()

You should see something like:

.. figure:: images/rvm_se.png
    :align: center

    The fit of RVM on the sample problem with SE basis centered on the
    data. Thre magenta disks show the relevant vectors that are finally
    kept. The blue symbols are the observed data. The red line is the
    true function. The blue line is predictive mean of the :ref:`glm`.
    The green lines are the borders of the 95% confidence interval
    about the mean.

Now, let's do the same problem with a :ref:`gpc` basis, orthogonal
with respect to a uniform distribution on :math:`[-10, 10]` and
of total degree 20::

    phi_gpc = best.gpc.OrthogonalPolynomials(20, left=-10, right=10)
    PHI_gpc = phi_gpc(X)
    rvm.set_data(PHI_gpc, Y)
    rvm.initialize()
    rvm.train()
    f_gpc = rvm.get_generalized_linear_model(phi_gpc)