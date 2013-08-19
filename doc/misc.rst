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