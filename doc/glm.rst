.. _glm:

Generalized Linear Model
========================

Let :math:`\left\{\phi_i(\cdot)\right\}_{i=1}^m` be a set of basis
functions (see :ref:`map-basis`). We think of a **Generalized Linear Model** (GLM) is a
parametrization of a subspace of the functions
:math:`\mathbf{f}:\mathbb{R}^d\rightarrow \mathbb{R}^q`:

    .. math::

        \mathbf{f}(\mathbf{x}; \mathbf{W}) =
        \boldsymbol{\phi}(\mathbf{x})^T\mathbf{W},

where :math:`\mathbf{W}\in\mathbb{R}^{m\times q}` is the weight matrix,
and

    .. math::

        \boldsymbol{\phi}(\mathbf{x}) =
        \left(\phi_1(\mathbf{x}), \dots, \phi_m(\mathbf{x})\right).

Usually, the weights :math:`\mathbf{W}` are not fixed, but its column
is has a multi-variate Gaussian distribution:

    .. math::

        \mathbf{W}_j \sim \mathcal{N}_m\left(\mathbf{W}_j |
        \mathbf{M}_j, \boldsymbol{\Sigma}\right),

for :math:`j=1,\dots,q`, where :math:`\mathbf{A}_j` is the :math:`j`-th
column of the matrix :math:`\mathbf{A}`, :math:`\mathbf{M}_j` is the mean
of :math:`\mathbf{M}_j` and semi-positive definite :math:`\boldsymbol{\Sigma}\in\mathbb{R}^{m\times m}`
mean of column :math:`j` and the covariance matrix, respectively.
Notice that we have restricted our attention to covariance
matrices independent of the output dimension. This is very restrictive
but in practice, there are ways around this problem. Giving a more
general definition would make it extremely difficult to store all
the required information (we would need a :math:`(qm)\times(qm)`
covariance matrix). In any case, this is the model we use in our
`RVM paper <http://epubs.siam.org/doi/pdf/10.1137/120861345>`_.

.. note::
    The distribution of the weights is to be thought as the posterior
    distribution for the weights that occures when you attempt to fit
    the model to some data.

Allowing for the possibility of some Gaussian noise, the predictive
distribution for the output :math:`\mathbf{y}` at the input point
:math:`\mathbf{x}` is given by:

    .. math::

        p(\mathbf{y} | \mathbf{x}) =
        \mathcal{N}_q\left(\mathbf{y} | \mathbf{m}(\mathbf{x}),
        \boldsymbol{\sigma}^2(\mathbf{x})\mathbf{I}_q\right),

where :math:`\mathbf{I}_q` is the :math:`q`-dimensional unit matrix,
while the mean and the variance at :math:`\mathbf{x}` are given by:

    .. math::

        \mathbf{m}(\mathbf{x}) = \boldsymbol{\phi}(\mathbf{x})^T
        \mathbf{W},\;\;
        \boldsymbol{\sigma}^2(\mathbf{x}) = \beta^{-1} +
        \boldsymbol{\phi}(\mathbf{x})^T\boldsymbol{\Sigma}
        \boldsymbol{\phi}(\mathbf{x}),

with :math:`\beta` being the noise precision (i.e., the inverse variance).

In BEST, we represent the GLM by a :class:`best.maps.GeneralizedLinearModel`
class which inherits from :class:`best.maps.Function`. It is essentially
a function that evaluates the predictive mean of the model.
However, it also offers access to several other useful methods for
uncertainty quantification.
Here is the definition of :class:`best.maps.GeneralizedLinearModel`:

.. class:: GeneralizedLinearModel

    A class that represents a Generalized Linear Model.

    .. method:: __init__(basis[, weights=None[, sigma_sqrt=None[, \
                         beta=None[, \
                         name='Generalized Linear Model']]]])

        Initialize the object.

        .. note::

            Notice that instead of the covariance matrix
            :math:`\boldsymbol{\Sigma}`, we initialize the object with
            its square root. The square root of
            :math:`\boldsymbol{\Sigma}` is any matrix
            :math:`\mathbf{R}\in \mathbb{R}^{k\times m}` such that:

                .. math::
                    \boldsymbol{\Sigma} = \mathbf{R}^T\mathbf{R}.

            This is usefull, because we allow for a the treatment of
            a semi-positive definite covariance (i.e., when
            :math:`k < m`). It is up to the user to supply the right
            :math:`\mathbf{R}` in there.

        :param basis: A set of basis functions.
        :type basis: :class:`best.maps.Function`
        :param weights: The mean weights \
                        :math:`\mathbf{M}`. If \
                        ``None``, then it is assumed to be all zeros.
        :type weights: 2D numpy array of shape :math:`m\times q`
        :param sigma_sqrt: The square root of the covariance materix. \
                           If ``None``, then it is assumed to be all \
                           zeros.
        :type sigma_sqrt: 2D numpy array of shape :math:`k\times q, k\le q`
        :param beta: The noise precision (inverse variance). If \
                     unspecified, it is assumed to be a very big \
                     number.
        :type beta: ``float``
        :param name: A name for the object.
        :type name: str

    .. method:: __call__(x)

        Evaluate the mean of the generalized model at ``x``.

        Essentially computed :math:`\mathbf{m}(\mathbf{x})`.

    .. method:: d(x)

        Evaluate the Jacobian of the generalized model at ``x``.

        This is :math:`\nabla \mathbf{m}(\mathbf{x})`.

    .. method:: get_predictive_covariance(x)

        Evaluate the predictive covariance at ``x``.

        Assume that ``x`` represents :math:`n` input points
        :math:`\left\{\mathbf{x}^{(i)})\right\}_{i=1}^n`.
        Then, this method computes the semi-positive definite matrix
        :math:`\mathbf{C}\in\mathbb{R}^n\times\mathbb{R}^n`, given by

            .. math::

                C_{ij} = \phi_k\left(\mathbf{x}^{(i)}\right)
                \Sigma_{kl}
                \phi_l\left(\mathbf{x}^{(j)}\right).

    .. method:: get_predictive_variance(x)

        Evaluate the predictive variance at ``x``.

        This is the diagonal of :math:`\mathbf{C}` of
        :func:`best.maps.GeneralizedLinearModel.get_predictive_covariance()`.
        However, it is computed without ever building :math:`\mathbf{C}`.

    .. attribute:: basis

        Get the underlying basis.

    .. attribute:: weights

        Get the weights.

    .. attribute:: sigma_sqrt

        Get the square root of the covariance matrix.

    .. attribute:: beta

        Get the inverse precision.