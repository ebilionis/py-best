.. _gpr::

Gaussian Process Regression
===========================

.. module:: best.gp
    :synopsis: Everything that is related to Gaussian Process Regression.

:note: The documentation of :mod:`best.gp` is incomplete at the moment \
       subject to change in the near future. The package includes lots \
       research details that are not fully documented here. The user \
       is adviced to use only what is discussed in this section and \
       look at the code itself in case he wants to do something more \
       excited. The code is at the moment not in a compete state.


.. _mgpr::

Multi-output Separable Gaussian Process Regression
--------------------------------------------------

At the moment, :ref:`best` does not offer a class for performing simple
Gaussian process regression (GPR) with 1D outputs.
Instead it offers a way to do
multi-output GP regression with a separable covariance function. The
simple case is a subcase of this more general case. We will not get
into the mathematical details of GPR. The user is adviced to consult:
    * Gaussian Process Regression book.
    * Multi-output Separable Gaussian Process Regression


The main class for performing GPR is
:class:`MultioutputGaussianProcess`. It can be trained either via
:ref:`mcmc` or :ref:`smc`. However, knowledge of these topics is not
required at this point since the object can be trained by itself.
The current version uses internally a :ref:`cov-se` with nuggets.
In future versions, we will let the user supply the :ref:`cov`.
Both the length scales of :ref:`cov-se` and the nuggets have
Exponential prior distributions with parameters that can be specified.
The mean of the model is assumed to be a :ref:`glm`. The user
must provide the design matrix.


.. class:: best.gp.MultioutputGaussianProcess

    :inherits: :class:`best.random.MarkovChainMonteCarlo`

    .. method:: __init__([mgp=None,[ name='MultioutputGaussianProcess']])

        Initialize the object.

        :param mgp: If supplied then the object copies all the data \
                    from here.
        :type mgp: :class:`best.gp.MultioutputGaussianProcess`

    .. set_data(X, H, Y)

        Set the observed data.

        :note: To see how the data should be organized for the separable \
               case, consult our paper.

        :param X: The input points. This must be a tuple of points \
                  observed for each separable dimension. E.g., if \
                  we have three separable dimensions, we must provide \
                  ``(X1, X2, X3)``.
        :type X: (tuple of) 2D numpy array(s)
        :param H: The design matrix.
        :type H: (tuple of) 2D numpy array(s)
        :param Y: The obverse outputs.
        :type Y: 2D numpy array

    .. method:: initialize(hyp[, eval=None])

        Inialize the object.

        :param hyp: The initial hyper-parameters. It must be a 1D \
                    numpy array ordered so that the first k elements \
                    correspond to the length scales and the last s to \
                    the nuggets. The length scales are ordered so that \
                    the first k[0] correspond to the first group of \
                    input variables, the following k[1] to the \
                    second and so on.
        :type hyp: 1D numpy array
        :param eval_state: A dictionary that contains all the data \
                           required to start the MCMC algorithm from \
                           the specified hyper-parameters. If not given \
                           then these data are initialized from scratch. \
                           The correct format of ``eval_state`` is the \
                           one returned by \
                           :func:`best.gp.MultioutputGaussianProcess.sample()`. \
                           So, do not try to improvise.

    .. method:: sample([x=None[, eval_state=None[, \
                        return_val_state=False[, steps=1]]])

        Take samples from the posterior of the hyper-parameters.

        :param x: The initial state. If not specified, we attemp to use \
                  the previous state processed by this class.
        :param eval_state: A dictionary containing the all the data \
                           required to initialize the object. Such a \
                           state is actually returned by this \
                           function if the option ``return_eval_sate`` \
                           is set to ``True``. If not specified, then \
                           everything is calculated from scratch.
        :param return_eval_state: If specified, then the routine returns  \
                                  the ``evaluated_state`` of the sampler, \
                                  which may be used to restart the MCMC \
                                  sampling.
        :returns: The current state of the MCMC (the hyper-parameters) \
                  and (optionally if ``return_eval_state``) is set \
                  all data required to continue the algorithm.

    .. method:: __call__(self, X, H[, Y=None[, C=None[, \
                                    compute_covariance=False]]])

        Evaluate the prediction at a given set of points.

        The result of this function, is basically the predictive
        distribution, encoded in terms of the mean ``Y`` and the
        covariance matrix ``C``.

        :param X: The input points.
        :param H: The design matrices.
        :param Y: An array to store the mean. If ``None``, then it is \
                  returned.
        :param C: An array to store the covariance. If ``None``, then
                  the \
                  covariance is not computed or it is returned as \
                  specified by the ``compute_covariance`` option.
        :param compute_covariance: If ``C`` is ``None``, and the flag \
                                   is set to ``True``, then the \
                                   covariance is calculated and \
                                   returned. If ``C`` is not ``None``, \
                                   then it is ignored.

    .. method:: sample_prediction(self, X, H[, Y=None[, C=None]])

        Sample from the predictive distribution of the model.

        :param X: The input points.
        :param H: The design matrices.
        :param Y: An array to store the mean. If ``None``, then it is \
                  returned.
        :param C: An optional array that will store the covariance \
                  matrix. If not supplied, it will be allocated. \
                  On the output, the incomplete Cholesky decomposition \
                  is written on ``C``.
        :returns: If ``Y`` is None, then the sample will be returned. \
                  The trace of the covariance normalized by the number \
                  of spatial/time inputs and the outputs. This is a \
                  measure associated with the uncertainty of the given \
                  input point.

    .. method:: add_data(self, X0, H0, Y0):

        Add more observations to the data set.

        The routine currently only adds observations pertaining to the \
        first component. Addition to the other components would ruin \
        the Kronecker properties of the matrices.

        :param X0: The input variables.
        :param H0: The design matrix.
        :param Y0: The observations.

    .. method:: sample_surrogate(self, X_design, H_design[, \
                                 rel_tol=0.1[, abs_tol=1e-3]])

        Sample a surrogate surface.

        Samples a surrogate surface that can be evaluated analytically. The
        procedure adds the design point with the maximum uncertainty defined
        by Eq. (19) of the paper and assuming a uniform input distribution
        until:
            + we run of design points,
            + or the <global> uncertainty satisfies a stopping criterion.
        The global uncertainty is defined to be the average uncertainty of
        all design points. The stopping criterion is implemented as follows:
            STOP if ``global uncertainty < rel_tol * init_unc or < abs_tol``,
        where init_unc is the initial uncertainty and rel_tol is a relative
        reduction and ``abs_tol`` is the absolute uncertainty we are willing to
        accept.

        :param X_design: The design points to be used. This
                         should be as dense as is computationally
                         feasible.
        :param rel_tol: We stop if the current uncertainty
                        is rel_tol times the initial uncertainty.
        :param abs_tol: We stop if the current uncertainty is
                        less than abs_tol.

    .. method:: evaluate_sparse(self, X, H[, compute_covariance=False[, \
                                sp_tol=0.1]])

        Evaluate the prediction at a given set of points.

        Same as
        :func:`best.gp.MultioutputGaussianProcess.__call__()`
        but we attemp to use sparse matrices.

    .. attribute:: sample_g

        Set/See if the nuggets are going to be sampled.

    .. attribute:: sample_r

        Set/See if the length scales are going to be sampled.

    .. attribute:: log_like

        The logarithm of the likelihood of the current state.

    .. attribute:: cov

        Get the covariance function.

    .. attribute:: num_mcmc

        The number of MCMC steps per Gibbs setp.

    .. attribute:: gamma

        Get the prior parameters for the length scales.

    .. attribute:: delta

        Get the prior parameters for the nuggets.

    .. attribute:: sigma_r

        Get the proposal step for the length scales.

    .. attribute:: sigma_g

        Get the proposal step for the nuggets.

    .. attribute:: g

        Get the current nuggets.

    .. attribute:: r

        Get the current length scales.

    .. attribute:: Sigma

        Get the output-correlation matrix.

    .. attribute:: log_post_lk

        Get the logarithm of the posterior likelihood.

    .. attribute:: acceptance_rate

        Get the MCMC acceptance rate.


A simple 1D example
-------------------

Typically, you would like to pick the hyper-parameters, observe the
convergence of :ref:`mcmc` or even use :ref:`smc` to train the model.
However, here is the simplest possible case we could run that works
just fine with the default parameters::

    import numpy as np
    import matplotlib.pyplot as plt
    from best.gp import MultioutputGaussianProcess
    # Number of observations
    num_obs = 20
    # The noise we will add to the data (std)
    noise = 1e-6
    # Draw the observed input points randomly
    X = -10. + 20. * np.random.rand(num_obs)
    X = np.atleast_2d(X).T
    # Draw the observations
    Y = np.sin(X) / (X + 1e-6) + noise * np.random.randn(*X.shape)
    # Construct the design matrix
    H = np.ones(X.shape)
    # Use RVM on the data
    gp = MultioutputGaussianProcess()
    gp.set_data(X, H, Y)
    # Pick the hyper-parameters (length scales, nuggets)
    hyp = np.array([1., 1e-6])
    gp.initialize(hyp)
    # Run 2000 MCMC steps
    gp.sample(steps=2000)
    # Get a function object (subject to change in the future)
    f = gp
    plt.plot(X, Y, '+', markersize=10)
    x = np.linspace(-10, 10, 100)
    x = np.atleast_2d(x).T
    h = np.ones(x.shape)
    fx, Cx = f(x, h, compute_covariance=True)
    plt.plot(x, fx, 'b', linewidth=2)
    plt.plot(x, np.sin(x) / (x + 1e-6), 'r', linewidth=2)
    s2 = 2. * np.sqrt(np.diag(Cx)).reshape(fx.shape)
    plt.plot(x, fx + s2, 'g')
    plt.plot(x, fx - s2, 'g')
    plt.show()

You should see something like the following figure:

    .. figure:: images/gp_1d.png
        :align: center

        The crosses are the observed data points. The red line is the
        true function from which the data are drawn. The blue line
        is the mean of the GPR prediction and the green lines indicated
        the 95% confidence intervals.


.. _tgpr:

Treed Gaussian Process Regression
---------------------------------

The class :class:`TreedMultioutputGaussianProcess` implements an
extension of the model we developed in (PAPER REFERENCE). This model
is not trained directly on data, but it requires a
:class:`best.maps.Solver` object. It is used to construct a surrogate
of the solver.

.. class best.gp.TreedMultioutputGaussianProcess

    .. method:: __init__(solver[, model=MultioutputGaussianProcess()[, \
                         mean_model=None[, \
                         tree=RandomElement(scale_X=True)]]])

            Initialize the object.

            :param solver: The solver you wish to learn.
            :type solver: :class:`best.maps.Solver`

    .. method:: initialize()

        Initialize the model.

    .. method:: train()

        Train the model to the solver.

    .. __call__(X, H, Y[, V=None])

        Evaluate the model at a particular point.


Simple Treed Gaussian Process Regression Example
------------------------------------------------

The following demo can be found in :file:`best/demo/test_treed_gp.py`.
It learns the output of a dynamical system with a discontinuity with
respect to the initial conditions (see :class:`examples.ko.KOSolver`).
It uses active learning (Bayesian Experimental Design) to select
the observed inputs::

    if __name__ == '__main__':
        import fix_path


    from examples.ko import KOSolver
    from best.gp import TreedMultioutputGaussianProcess
    import numpy as np
    import matplotlib.pyplot as plt


    if __name__ == '__main__':
        # Initialize the solver
        solver = KOSolver(k=2, T=[0, 1], n_t=32)
        # Initialize the treed GP
        tmgp = TreedMultioutputGaussianProcess(solver=solver)
        tmgp.num_xi_init = 10
        tmgp.num_xi_test = 100
        tmgp.num_max = 100
        tmgp.num_elm_max = 20
        tmgp.verbose = True
        tmgp.model.sample_g = True
        tmgp.model.num_mcmc = 1
        tmgp.model.num_init = 100
        # Initialial hyper-parameters
        init_hyp = np.array([.1, .1, .1, 1e-1, 1e-1])
        tmgp.init_hyp = init_hyp
        tmgp.num_mcmc = 100
        # Train
        tmgp.train()
        # Print the tree
        print str(tmgp.tree)
        # A fine scale solver to test our predictions
        fine_solver = KOSolver(k=solver.k_of[0], n_t=50)
        # Make predictions
        for i in range(10):
            xi = np.random.rand(1, solver.k_of[0])
            X = [xi] + fine_solver.X_fixed
            H = tmgp.mean_model(X)
            n = np.prod([x.shape[0] for x in X])
            Yp = np.ndarray((n, solver.q), order='F')
            Vp = np.ndarray((n, solver.q), order='F')
            tmgp(X, H, Yp, Vp)
            Y = fine_solver(xi[0, :])
            plt.plot(fine_solver.X_fixed[0], Y)
            E = 2. * np.sqrt(Vp)
            for i in range(solver.q):
                plt.errorbar(fine_solver.X_fixed[0], Yp[:, i], yerr=E[:, i])
            plt.show()

The plots you will see will look like the following:

.. figure:: images/tgp_ko.png
    :align: center

    The prediction of the treed Gaussian Process model for the response
    of the dynamical system as a function of time with error bars. This
    the prediction on a random input sample not used in the training
    data. The total number of observations was restricted to 100.