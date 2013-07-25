"""A separable function wrapper.

Author:
    Ilias Bilionis

Date:
    2/3/2013
"""

from uq.maps import Function
from uq.maps import ConstantFunction
import numpy as np


class SeparableFunctionWrapper(Function):
    """A separable function wrapper.

    This class serves as a wrapper of a separable function of multiple input
    components. It can basically screen some or all of these components.
    """

    # The inner separable function
    _sf = None

    # The observed inputs
    _X_obs = None

    # The observed basis functions
    _H_obs = None

    # The range of outputs
    _y_idx = None

    # A temporary place to store the output of the inner function.
    _Y_all = None

    # The mean function
    _mean_func = None

    @property
    def sf(self):
        """Get the inner separable function."""
        return self._sf

    @property
    def X_obs(self):
        """Get the observed inputs."""
        return self._X_obs

    @property
    def H_obs(self):
        """Get the observed basis functions."""
        return self._H_obs

    @property
    def y_idx(self):
        """Get the observed output indices."""
        return self._y_idx

    @property
    def mean_func(self):
        """Get the mean function."""
        return self._mean_func

    @property
    def num_obs(self):
        """Get the number of observed inputs."""
        return 1 if self.X_obs is None else self.X_obs.shape[0]

    def __init__(self, sf, X_obs=None, H_obs=None, y_range=':',
            mean_func=None):
        """Initialize the object.

        Arguments:
            sf      ---     The inner separable function.

        Keyword Arguments:
            X_obs   ---     The observed inputs. If None, then we assume that
                            sf has only one input component.
            H_obs   ---     The observed design matrix. If None, then we
                            assume that they don't exist or that they are all
                            one.
            y_range ---     The range of outputs that are observed. Any python
                            indexing convention is valid. The default is that
                            all the outputs are observed.
            mean_func ---   The meanfunction of the surrogate. If None, then
                            an all ones function is assumed.
        """
        assert hasattr(sf, '__call__')
        self._sf = sf
        if X_obs is None:
            # Assuming that sf has only one component
            assert self.sf.s == 1
        else:
            assert isinstance(X_obs, np.ndarray)
            assert X_obs.ndim <= 2
            if X_obs.ndim == 1:
                X_obs = X_obs.reshape((X_obs.shape[0], 1))
            k_rest = np.prod(self.sf.k_of[1:])
            # Asserting the input dimensions match
            assert k_rest == X_obs.shape[1]
        self._X_obs = X_obs
        if H_obs is None:
            if self.X_obs is not None:
                H_obs = np.ones((self.X_obs.shape[0], 1))
        if H_obs is not None:
            assert isinstance(H_obs, np.ndarray)
            assert H_obs.ndim <= 2
            if H_obs.ndim == 1:
                H_obs = H_obs.reshape((H_obs.shape[0], 1))
            m_rest = np.prod(sf.m_of[1:])
            assert m_rest == H_obs.shape[1]
            assert self.X_obs.shape[0] == H_obs.shape[0]
        self._H_obs = H_obs
        if mean_func is None:
            mean_func = ConstantFunction(self.sf.k_of[0], 1.)
        else:
            assert isinstance(mean_func, Function)
            assert mean_func.num_input == self.sf.k_of[0]
            assert mean_func.num_output == self.sf.m_of[0]
        self._mean_func = mean_func
        assert isinstance(y_range, str)
        self._y_idx = range(self.sf.q)
        self._y_idx = eval('self._y_idx[' + y_range + ']')
        num_output = len(self._y_idx)
        num_output *= 1 if self.X_obs is None else self.X_obs.shape[0]
        self._Y_all = np.ndarray((np.prod(self.sf.n_of[1:]), self.sf.q))
        super(SeparableFunctionWrapper, self).__init__(self.sf.k_of[0],
                num_output,
                name='Separable Function Wrapper')

    def __call__(self, x, y=None):
        """Evaluate the function at x."""
        assert isinstance(x, np.ndarray)
        assert x.ndim <= 2
        if x.ndim == 1:
            x = x.reshape((1, x.shape[0]))
        assert x.shape[1] == self.sf.k_of[0]
        h = self.mean_func(x)
        if h.ndim == 1:
            h = h.reshape((1, h.shape[0]))
        y = []
        for i in range(self.num_obs):
            X = (x, )
            H = (h, )
            c_k = 0
            c_m = 0
            for j in range(1, self.sf.s):
                X += (self.X_obs[i:(i+1), c_k:(c_k + self.sf.k_of[j])], )
                c_k += self.sf.k_of[j]
                H += (self.H_obs[i:(i+1), c_m:(c_m + self.sf.m_of[j])], )
                c_m += self.sf.m_of[j]
            y.append(self.sf(X, H)[:, self._y_idx])
        return np.vstack(y).flatten(order='F')
