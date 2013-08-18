.. _rvm:

Relevance Vector Machine
========================

Relevance Vector Machine (RVM) trains a :ref:`glm`
yielding sparse representation (i.e., many of the basis functions are
not used at the end). The implementation in BEST is the
Multi-dimensional Relevance Vector Machine (MRVM) as described in
our `paper <http://epubs.siam.org/doi/pdf/10.1137/120861345>`_.
It uses the :ref:`linalg-gsvd` to train the
model, which is considerably more stable than relying to Cholesky
decompositions. This is achieved via the
:class:`best.linalg.GeneralizedSVD` class.

As already said, we start with a GLM

.. figure:: images/rvm_se.png
    :align: center

    The fit of RVM on the sample problem with SE basis centered on the
    data.