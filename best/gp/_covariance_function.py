"""Define the base covariance function class.

Author:
    Ilias Bilionis

Date:
    11/20/2012

"""


__all__ = ['CovarianceFunction']


from .. import Object


class CovarianceFunction(Object):

    """Base covariance function class."""

    # The name of the covariance function
    _name = None

    def _check_hyp(self, hyp):
        """Check if the arguments are valid.

        It should be implemented by the deriving classes.
        The implemenations should raise
        a TypeError if the type of hyp is wrong or and ValueError if the values
        are wrong.
        The default behavior of the CovarianceFunction class it to accept every
        input.

        """
        pass

    def __init__(self, name="CovarianceFunction"):
        """Initialize the class.

        Keyword Arguments:
        name    ---     The name of the covariance function.
        """
        super(CovarianceFunction, self).__init__(name=name)

    def __call__(self, hyp, x1, x2=None, A=None):
        """Compute the covariance matrix.

        This function must be implemented by all deriving classes.
        The inputs hyp and x1 must always be provided. If x2 is not provided,
        then the self covariance matrix is computed. If A is not provided, then
        the matrix should be allocated and returned.

        Arguments:
        hyp     ---     Hyper parameters.
        x1      ---     Input list of inputs.

        Keyword Arguments:
        x2      ---     Input list of inputs.
        A       ---     Covariance matrix.

        """
        raise NotImplementedError(
                'The __call__ of CovarianceFunction is not implemented.')
