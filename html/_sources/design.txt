.. _design:

Design of Experiments
=====================

.. module:: best.design
    :synopsis: Perform design of experiments.


This module implements functionality related to the design of
(computer) experiments.


Latin Hyper-cube Designs
------------------------

Latin hyper-cube designs approximate samples from a uniform distribution
attempting to keep the degeneracy of the samples low.

.. function:: lhs(n, k[, seed=lhs_seed()])

    Fill an :math:`n\times k` matrix with a latin hyper-cube design.

    :param n: The number of samples.
    :type n: int
    :param k: The number of dimensions.
    :type k: int
    :param seed: A random seed. If not specified then it is taken from \
                :func:`best.design.lhs_seed()`.
    :type seed: int
    :returns: A latin hyper-cube design.
    :rtype: 2D numpy array.

    Here is an example::

        from best.design import lhs
        X = lhs(10, 2)
        print X

    This should print something similar to::

        [[ 0.55  0.65]
         [ 0.25  0.75]
         [ 0.85  0.95]
         [ 0.05  0.45]
         [ 0.15  0.35]
         [ 0.65  0.45]
         [ 0.25  0.15]
         [ 0.05  0.75]
         [ 0.35  0.55]
         [ 0.85  0.95]]


.. function:: lhs_seed()

    Produce a random seed to be used in :func:`best.design.lhs()`.

    :returns: A random seed.
    :rtype: int
