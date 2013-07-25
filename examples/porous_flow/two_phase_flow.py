"""A simple finite-volume solver for two-phase immiscible flow.

We solve the system of equations:

    -div(K l(s) grad(p)) = q,   (1)
    ds/dt + div(f(s)u) = q_s,   (2)

in a 3D domain, where:
    + K is the permeability tensor (assume to be diagonal for simplicity).
    + p the capillary pressure (p_w - p_o, where p_w is the water pressure
    and p_o the oil pressure).
    + u = -K l(s) grad(p) = u_w + u_v: is the total velocity.
    + s = s_w = 1. - s_o: is the water saturation.
    + q is the source term of the pressure which models injection and
    production wells.
    + l(s) = l_w(s) + l_o(s): is the total mobility.
    + l_w(s) = k_rw/m_rw: is the water mobility.
    + l_o(s) = k_ro/m_ro: is the oil mobility.
    + k_rw is the relative permeability of water in oil.
    + k_ro is the relative permeability of oil in water.
    + m_rw is the viscosity of water.
    + m_ro is the viscosity of oil.
    + f(s) = l_w(s) / l(s): is the fractional flow function.
    + q_s = max(q, 0) + f(s)min(q, 0) is the source term for the saturation.
    + t is time.

We use non-flux boundary conditions.

Author:
    Ilias Bilionis

Date:
    2/11/2013

"""


import numpy as np
import scipy.sparse as sp
import math
#import matplotlib.pyplot as plt


class TwoPhaseFlow(object):
    """A two-phase immiscible flow solver."""

    # Number of grid blocks per dimension (tuple)
    _n = None

    # Total number of grid blocks
    _n_all = None

    # The grid
    _grid = None

    # The length of the domain along each dimension (tuple)
    _L = None

    # The step size along each dimension (tuple)
    _h = None

    # The connate water saturation
    _s_wc = None

    # The irreducible oil saturation
    _s_or = None

    # The water viscosity
    _v_w = None

    # The oil viscosity
    _v_o = None

    # The volume of the cells
    _V = None

    # The porosity
    _por = None

    # The time step
    _dt = None

    @property
    def n(self):
        """Get the number of gridblocks per dimension."""
        return self._n

    @property
    def n_all(self):
        """Get the total number of gridblocks."""
        return self._n_all

    @property
    def grid(self):
        """Get the grid."""
        return self._grid

    @property
    def L(self):
        """Get the size per dimension."""
        return self._L

    @property
    def h(self):
        """Get the step size per dimension."""
        return self._h

    @property
    def s_wc(self):
        """Get connate water saturation."""
        return self._s_wc

    @property
    def s_or(self):
        """Get irreducible oil saturation."""
        return self._s_or

    @property
    def v_w(self):
        """Get the water viscosity."""
        return self._v_w

    @property
    def v_o(self):
        """Get the oil viscosity."""
        return self._v_o

    @property
    def V(self):
        """Get the volume of the cells."""
        return self._V

    @property
    def por(self):
        """Get the porosity."""
        return self._por

    @property
    def dt(self):
        """The desired time step."""
        return self._dt

    @dt.setter
    def dt(self, value):
        """Set the time step."""
        assert isinstance(value, float)
        assert value > 0.
        self._dt = value

    def get_closest_idx(self, X):
        """For each element in X, get the index of the closest grid point."""
        X = np.array(X)
        X_all = self.grid.reshape((3, self.n_all), order='F').T
        idx = []
        for i in range(X.shape[0]):
            idx.append(np.argmin(np.sum((X_all - X[i, :]) ** 2, axis=1)))
        return idx

    def __init__(self, nx=64, ny=64, nz=1, Lx=1., Ly=1., Lz=1.,
            s_wc=0., s_or=0., v_w=1., v_o=1., por=1.,
            dt=0.1):
        """Initialize the object."""
        self._n = (nx, ny, nz)
        self._n_all = np.prod(self.n)
        self._grid = np.mgrid[0:Lx:(nx * 1j), 0:Ly:(ny * 1j), 0:Lz:(nz * 1j)]
        self._L = (Lx, Ly, Lz)
        self._h = (Lx / nx, Ly / ny, Lz / nz)
        self._s_wc = s_wc
        self._s_or = s_or
        self._v_w = v_w
        self._v_o = v_o
        self._V = np.array([np.prod(self.h)])
        if isinstance(por, np.ndarray):
            self._por = por
        else:
            self._por = por * np.ones(self.n)
        self.dt = dt

    def _tpfa(self, K0, K, q):
        """Finite volume discretization of -div(K * grad(u)) = q."""
        # Compute transmissibilities by harmonic averaging.
        nx = self.n[0]
        ny = self.n[1]
        nz = self.n[2]
        n = self.n_all
        hx = self.h[0]
        hy = self.h[1]
        hz = self.h[2]
        L = 1. / K
        tx= 2. * hy * hz / hx
        TX = np.zeros((nx + 1, ny, nz))
        ty = 2. * hx * hz/ hy
        TY = np.zeros((nx, ny + 1, nz))
        tz = 2. * hx * hy / hz
        TZ = np.zeros((nx, ny, nz + 1))
        TX[1:nx, :, :] = tx / (L[0, :(nx - 1), :, :] + L[0, 1:nx, :, :])
        TY[:, 1:ny, :] = ty / (L[1, :, :(ny - 1), :] + L[1, :, 1:ny, :])
        TZ[:, :, 1:nz] = tz / (L[2, :, :, :(nz -1)] + L[2, :, :, 1:nz])
        # Assemble TPFA discretization matrix.
        x1 = np.reshape(TX[:nx, :, :], (n, 1), order='F')
        x2 = np.reshape(TX[1:(nx + 1), :, :], (n, 1), order='F')
        y1 = np.reshape(TY[:, :ny, :], (n, 1), order='F')
        y2 = np.reshape(TY[:, 1:(ny + 1), :], (n, 1), order='F')
        z1 = np.reshape(TZ[:, :, :nz], (n, 1), order='F')
        z2 = np.reshape(TZ[:, :, 1:(nz + 1)], (n, 1), order='F')
        diag_vecs = np.hstack([-z2, -y2, -x2, x1 + x2 + y1 + y2 + z1 + z2,
                           -x1, -y1, -z1])
        diag_indx = np.array([-nx * ny, -nx, -1, 0, 1, nx, nx * ny],
                dtype='i')
        A = sp.spdiags(diag_vecs.T, diag_indx, n, n, format='csc')
        A[0, 0] += np.sum(K0[:, 0, 0, 0])
        # Solve the linear system and extract interface fluxes.
        u = sp.linalg.spsolve(A, q)
        #u, info = sp.linalg.cg(A, q)
        #u, info = sp.linalg.gmres(A, q)
        P = np.reshape(u, (nx, ny, nz), order='F')
        Vx = np.zeros((nx + 1, ny, nz))
        Vy = np.zeros((nx, ny + 1, nz))
        Vz = np.zeros((nx, ny, nz + 1))
        Vx[1:nx, :, :] = (P[:(nx -1 ), :, :] - P[1:nx, :, :]) * TX[1:nx, :, :]
        Vy[:, 1:ny, :] = (P[:, :(ny - 1), :] - P[:, 1:ny, :]) * TY[:, 1:ny, :]
        Vz[:, :, 1:nz] = (P[:, :, :(nz - 1)] - P[:, :, 1:nz]) * TZ[:, :, 1:nz]
        return P, [Vx, Vy, Vz]

    def _rel_perm(self, s, compute_der=False):
        """Relative permeabilities of oil and water."""
        ss = (s - self.s_wc) / (1. - self.s_wc - self.s_or)
        m_w = ss ** 2 / self.v_w
        m_o = (1. - ss) ** 2 / self.v_o
        if not compute_der:
            return m_w, m_o
        else:
            d_m_w = 2. * ss / self.v_w / (1. - self.s_wc - self.s_or)
            d_m_o = -2. * (1. - ss) / self.v_o / (1. - self.s_wc - self.s_or)
            return m_w, m_o, d_m_w, d_m_o

    def _pres(self, K, s, q):
        """Finite volume discretization of -div(K * l(s) * grad(u)) = q."""
        m_w, m_o = self._rel_perm(s)
        m_t = m_w + m_o
        KM = np.reshape(np.hstack([m_t, m_t, m_t]).T, (3, ) + self.n,
                order='F') * K
        return self._tpfa(K, KM, q)

    def gen_A(self, v, q):
        """Matrix assembly in upwind finite-volume discretization of saturation eq."""
        nx = self.n[0]
        ny = self.n[1]
        nz = self.n[2]
        n = self.n_all
        # Production
        fp = np.minimum(q, 0.)
        # Separe flux into
        # - flow in positive coordinate direction (XP, YP, ZP)
        # - flow in negative coordinate direction (XN, YN, ZN)
        XN = np.minimum(v[0], 0.)
        x1 = np.reshape(XN[:nx, :, :], (n, 1), order='F')
        YN = np.minimum(v[1], 0.)
        y1 = np.reshape(YN[:, :ny, :], (n, 1), order='F')
        ZN = np.minimum(v[2], 0.)
        z1 = np.reshape(ZN[:, :, :nz], (n, 1), order='F')
        XP = np.maximum(v[0], 0.)
        x2 = np.reshape(XP[1:(nx + 1), :, :], (n, 1), order='F')
        YP = np.maximum(v[1], 0.)
        y2 = np.reshape(YP[:, 1:(ny + 1), :], (n, 1), order='F')
        ZP = np.maximum(v[2], 0.)
        z2 = np.reshape(ZP[:, :, 1:(nz + 1)], (n, 1), order='F')
        diag_vecs = np.hstack([z2, y2, x2, fp + x1 - x2 + y1 - y2 + z1 - z2,
                               -x1, -y1, -z1])
        diag_indx = np.array([-nx * ny, -nx, -1, 0, 1, nx, nx * ny])
        A = sp.spdiags(diag_vecs.T, diag_indx, n, n, format='csr')
        return A

    def _newt_raph(self, s, v, q, T):
        """Implicit scheme for the saturation equation."""
        n = self.n_all
        A = self.gen_A(v, q)
        conv = 0.
        IT = 0.
        s00 = s
        while conv == 0:
            # Time step
            dt = T / (2 ** IT)
            #print 'dt, ', dt
            # Time step / pore volume
            dtx = dt / (self.V.ravel(order='F')
                    * self.por.ravel(order='F')).reshape((n, 1), order='F')
            fi = np.maximum(q, 0) * dtx
            B = sp.spdiags(dtx.T, 0, n, n, 'csr') * A

            # Loop over sub-timesteps
            I = 0
            while I < 2 ** IT:
                s0 = s
                dsn = 1
                it = 0
                I += 1
                # ITERATION
                while dsn > 1e-3 and it < 10:
                    # Mobilities and derivatives
                    m_w, m_o, d_m_w, d_m_o = self._rel_perm(s,
                            compute_der=True)
                    # df_w / ds
                    df = d_m_w / (m_w + m_o) - m_w / (m_w + m_o) ** 2. * (d_m_w + d_m_o)
                    # G'(s)
                    dG = sp.eye(n, n) - B * sp.spdiags(df.T, 0, n, n)
                    # Fractional flow
                    f_w = m_w / (m_w + m_o)
                    # G(s)
                    G = s - s0 - ((B * f_w) + fi)
                    # Incremen ds
                    ds = sp.linalg.spsolve(dG, -G).reshape((n, 1), order='F')
                    s = s + ds
                    dsn = np.linalg.norm(ds)
                    it += 1
                if dsn > 1e-3:
                    I = 2 ** IT
                    s = s00
            if dsn < 1e-3:
                conv = 1
            else:
                IT += 1
        return s

    def solve(self, K, T, X, IR, Pt=10., St=5.):
        """Solve the problem for a given permeability tensor and source q"""
        #print 'Solving'
        X = np.array(X)
        X[:, 0] *= self.L[0]
        X[:, 1] *= self.L[1]
        X[:, 2] *= self.L[2]
        idx = self.get_closest_idx(X)
        #print idx
        q = np.zeros((self.n_all, 1))
        q[idx[0]] = IR
        q[idx[1:]] = -IR / (X.shape[0] - 1.)
        #print IR
        # Initial saturation
        s = np.ones((self.n_all, 1)) * self.s_wc
        m_o_all = []
        t_all = []
        #fig = plt.figure()
        #ax = fig.add_subplot(111)
        #ax.set_aspect('equal')
        #p, v = self._pres(K, s, q)
        for tp in range(int(math.ceil(T / Pt))):
            p, v = self._pres(K, s, q)
            for ts in range(int(math.ceil(Pt / St))):
                #print ts
                s = self._newt_raph(s, v, q, self.dt)
            #if tp == 0:
            #    continue
            t_all.append(tp * Pt)
            #print 'time: ', tp * Pt
            m_o_a = []
            for i in idx:
                m_w, m_o = self._rel_perm(s[i])
                m_t = m_w + m_o
                m_o_a.append(m_o / m_t)
            m_o_all.append(m_o_a)
            #c = ax.contourf(s.reshape(self.n, order='F')[:, :, 0].T,
            #        extent=[0, self.L[0], 0, self.L[1]])
            #plt.colorbar(c)
            #plt.show(block=False)
            #plt.draw()
            #plt.savefig('spe10/sat_' + str(tp) + '.png')
        return np.array(t_all), np.array(m_o_all)[:, :, 0]