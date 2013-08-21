"""The base class for all distributions.

Author:
    Ilias Bilionis

Date:
    1/14/2013

"""


__all__ = ['Distribution']


from . import LikelihoodFunction


class Distribution(LikelihoodFunction):
    """The base class of all distributions."""

    def __init__(self, num_input, name='Distribution'):
        """Initialize the object.

        Arguments:
            num_input   ---     The number of input dimensions.

        Keyword Arguments:
            name        ---     A name for this distribution.
        """
        super(Distribution, self).__init__(num_input, 1, name=name)

    def sample(self, x=None):
        """Sample the distribution.

        Keyword Arguments:
        x   ---     If it is specified then x should be overriden.
                    If it is not specified, then the sample is allocated and
                    returned.
        """
        raise NotImplementedError('Must be overriden by deriving classes.')