.. _misc:

Miscellaneous
=============

    .. module:: best.misc
        :synopsis: This module contains functions that do not fit \
                   anywhere else.

The module :mod:`best.misc` contains functions that do not fit
anywhere else in :mod:`best`.


.. function:: logsumexp(a)

    Computes in a stable way the following expression for the array
    ``a``:

        .. math::
            M = \log\sum_i e^{a_i}.

    It uses the `log-sum-exp trick <http://machineintelligence.tumblr.com/post/4998477107/the-log-sum-exp-trick>`_.

    :param a: An array.
    :type a: 1D numpy array.
    :returns: :math:`M` as defined above.
    :rtype: float


.. function:: multinomial_resample(p)

    This functions accepts a probability distribution :math:`p_i` over
    :math:`{0, 1, \dots, n}`.
    This function returns the result of sampling :math:`n` times from
    this distribution.

    :param p: An array of positive numbers.
    :type p: 1D numpy array
    :returns: An array of integers tween 0 and ``p.shape[0] - 1``.
    :rtype: 1D numpy array of ``int``.

    Here is a small example::

        from best.misc import multinomial_resample
        p = [0.25, 0.25, 0.25, 0.25]
        multinomial_resample(p)

    This should print something like::

        array([0, 2, 2, 3], dtype=int32)