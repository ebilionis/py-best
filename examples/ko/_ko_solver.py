"""A solver for the KO problem.

Author:
    Ilias Bilionis

Date:
    12/2/2012

"""


__all__ = ['KOSolver', 'SingleKOSolver']


from best.maps import Solver
import numpy as np
from scipy.integrate import ode


class KOSolver(Solver):
    """A solver for the KO-problem."""

    def __init__(self, k=1, T=[0, 1.], n_t=10):
        """Initialize the object.

        Keyword Arguments:
            k   ---  Number of input dimensions.
            T   ---  The time domain.
            n_t  ---  The number of splits in the time domain.

        """
        super(KOSolver, self).__init__(s=2, q=3)
        self.n_of_fixed[0] = n_t
        self.X_fixed[0] = np.linspace(T[0], T[1], n_t).reshape((n_t, 1))
        self.k_of[0] = k
        self.k_of[1] = 1
        def f(t, y):
            return [y[0] * y[2], - y[1] * y[2],
                    - y[0] * y[0] + y[1] * y[1]]
        def jac(t, y):
            return [[y[2], 0., y[0]],
                    [0., -y[2], -y[1]],
                    [-2. * y[0], 2. * y[1], 0.]]
        self._f = f
        self._jac = jac
        self._ode = ode(self._f, jac=self._jac)
        self._ode.set_integrator('vode', method='bdf', with_jacobian=True,
                max_step=1e-2)

    def __call__(self, xi, Y=None):
        """Evaluate the solver.

        Arguments:
            xi  ---     An array of dimension self.k_of[0] with elements in
                        [0, 1].

        Keyword Arguments:
            Y   ---     The output array. It must be of dimension
                        self.n_of_fixed[0] x self.q. If it not provided, then
                        it is allocated and returned.

        """
        return_Y = False
        if Y is None:
            return_Y = True
            Y = np.ndarray((self.n_of_fixed[0], self.q), order='F')
        if self.k_of[0] == 1:
            y0 = [1., 0.1 * (2. * xi[0] - 1.), 0.]
        elif self.k_of[0] == 2:
            y0 = [1., 0.1 * ( 2. * xi[0] - 1.), 2. * xi[1] - 1.]
        elif self.k_of[0] == 3:
            y0 = [2. * xi[0] - 1, 2. * xi[1] - 1, 2. * xi[2] - 1]
        t0 = self.X_fixed[0][0, 0] * 10.
        self._ode.set_initial_value(y0, t0)
        Y[0, :] = y0
        count = 1
        while self._ode.successful() and count < self.n_of_fixed[0]:
            t = self.X_fixed[0][count, 0] * 10.
            self._ode.integrate(t)
            Y[count, :] = self._ode.y
            count += 1
        if return_Y:
            return Y


class SingleKOSolver(Solver):
    """An alternative KO solver."""

    def __init__(self, k=1, T=[0, 1.], n_t=10):
        super(SingleKOSolver, self).__init__(s=1, q=(n_t * 3))
        self.t = np.linspace(T[0], T[1], n_t)
        self.n_t = n_t
        self.k_of[0] = k
        def f(t, y):
            return [y[0] * y[2], - y[1] * y[2],
                    - y[0] * y[0] + y[1] * y[1]]
        def jac(t, y):
            return [[y[2], 0., y[0]],
                    [0., -y[2], -y[1]],
                    [-2. * y[0], 2. * y[1], 0.]]
        self._f = f
        self._jac = jac
        self._ode = ode(self._f, jac=self._jac)
        self._ode.set_integrator('vode', method='bdf', with_jacobian=True,
                max_step=1e-2)

    def __call__(self, xi, Y=None):
        """Evaluate the solver.

        Arguments:
            xi  ---     An array of dimension self.k_of[0] with elements in
                        [0, 1].

        Keyword Arguments:
            Y   ---     The output array. It must be of dimension
                        self.n_of_fixed[0] x self.q. If it not provided, then
                        it is allocated and returned.

        """
        return_Y = False
        if Y is None:
            return_Y = True
            Y = np.ndarray((1, self.q), order='F')
        Y = Y.reshape((self.n_t, 3), order='F')
        if self.k_of[0] == 1:
            y0 = [1., 0.1 * (2. * xi[0] - 1.), 0.]
        elif self.k_of[0] == 2:
            y0 = [1., 0.1 * ( 2. * xi[0] - 1.), 2. * xi[1] - 1.]
        elif self.k_of[0] == 3:
            y0 = [2. * xi[0] - 1, 2. * xi[1] - 1, 2. * xi[2] - 1]
        #t0 = self.X_fixed[0][0, 0] * 10.
        t0 = self.t[0]
        self._ode.set_initial_value(y0, t0)
        Y[0, :] = y0
        count = 1
        while self._ode.successful() and count < self.n_t:
            #t = self.X_fixed[0][count, 0] * 10.
            t = self.t[count]
            self._ode.integrate(t)
            Y[count, :] = self._ode.y
            count += 1
        if return_Y:
            return Y.reshape((1, self.q), order='F')