"""The base class representing a continuous Markov chain.

Author:
    Ilias Bilionis

Date:
    1/15/2013

"""


__all__ = ['MarkovChain']


class MarkovChain(object):
    """The base class of a Markov chain."""

    def __init__(self):
        """Initializes the object."""
        pass

    def __call__(x_p, x_n):
        """Evaluates the logarithm of the pdf of the chain.

        Usually this would be written as:
            log p(x_n | x_p),
        but we are using the programming convention that whatever
        is given comes first.

        Arguments:
        x_p     ---     The state on which we are conditioning.
        x_n     ---     The new state.

        Return:
        The pdf of the state.
        """
        raise NotImplementedError('This needs to be overriden by the deriving class.')

    def sample(x_p, x_n):
        """Sample from the Markov chain and write the result on x_n.

        Arguments:
        x_p     ---     The state on which we are conditioning.
        x_n     ---     The new state. To be overriden.
        """
        raise NotImplementedError('This needs to be overriden by the deriving class.')