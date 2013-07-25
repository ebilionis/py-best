from fipy import *
import numpy as np
import scipy.sparse as sp
import math
import matplotlib.pyplot as plt
from uq.gp import *
from uq.random import *


def tpfa(mesh, K, q):
    """Finite volume discretization of -div(K * grad(u)) = q.
    
    Arguments:
        grid    ---     A finite volume mesh.
        K       ---     The permeability.
        q       ---     The source term.
    """
    # Compute transmissibilities by harmonic averaging.
    nx = mesh.nx
    ny = mesh.ny
    nz = mesh.nz
    n = nx * ny * nz
    hx = mesh.hx
    hy = mesh.hy
    hz = mesh.hz
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
    x1 = np.reshape(TX[:nx, :, :], (n, 1), order='F').copy()
    x2 = np.reshape(TX[1:(nx + 1), :, :], (n, 1), order='F').copy()
    y1 = np.reshape(TY[:, :ny, :], (n, 1), order='F').copy()
    y2 = np.reshape(TY[:, 1:(ny + 1), :], (n, 1), order='F').copy()
    z1 = np.reshape(TZ[:, :, :nz], (n, 1), order='F').copy()
    z2 = np.reshape(TZ[:, :, 1:(nz + 1)], (n, 1), order='F').copy()
    #x1 = TX[:nx, :, :].ravel(order='F')
    #x2 = TX[1:(nx + 1), :, :].ravel(order='F')
    #y1 = TY[:, :ny, :].ravel(order='F')
    #y2 = TY[:, 1:(ny + 1), :].ravel(order='F')
    #z1 = TZ[:, :, :nz].ravel(order='F')
    #z2 = TZ[:, :, 1:(nz + 1)].ravel(order='F')
    #print z2.copy().flags
    #quit()
    diag_vecs = np.hstack([-z2, -y2, -x2, x1 + x2 + y1 + y2 + z1 + z2,
                           -x1, -y1, -z1]).copy()
    #diag_vecs = np.concatenate([-z2, -y2, -z2, x1 + x2 + y1 + y2 + z1 + z2,
    #                            -x1, -y1, -z1])
    #diag_vecs = [-z2, -y2, -x2, x1 + x2 + y1 + y2 + z1 + z2, -x1, -y1, -z1]
    diag_indx = np.array([-nx * ny, -nx, -1, 0, 1, nx, nx * ny], dtype='i')
    A = sp.spdiags(diag_vecs.T, diag_indx, n, n, format='csc')
    A[0, 0] += np.sum(mesh.K[:, 0, 0, 0])
    
    # Solve the linear system and extract interface fluxes.
    u = sp.linalg.spsolve(A, q)
    #u, info = sp.linalg.cg(A, q)
    #u, info = sp.linalg.gmres(A, q)
    #print info
    P = np.reshape(u, (nx, ny, nz), order='F')
    Vx = np.zeros((nx + 1, ny, nz))
    Vy = np.zeros((nx, ny + 1, nz))
    Vz = np.zeros((nx, ny, nz + 1))
    Vx[1:nx, :, :] = (P[:(nx -1 ), :, :] - P[1:nx, :, :]) * TX[1:nx, :, :]
    Vy[:, 1:ny, :] = (P[:, :(ny - 1), :] - P[:, 1:ny, :]) * TY[:, 1:ny, :]
    Vz[:, :, 1:nz] = (P[:, :, :(nz - 1)] - P[:, :, 1:nz]) * TZ[:, :, 1:nz]
    return P, [Vx, Vy, Vz]

def rel_perm(s, fluid, compute_der=False):
    """Relative permeabilities of oil and water."""
    ss = (s - fluid.s_wc) / (1. - fluid.s_wc - fluid.s_or)
    m_w = ss ** 2 / fluid.v_w
    m_o = (1. - ss) ** 2 / fluid.v_o
    if not compute_der:
        return m_w, m_o
    else:
        d_m_w = 2. * ss / fluid.v_w / (1. - fluid.s_wc - fluid.s_or)
        d_m_o = -2. * (1. - ss) / fluid.v_o / (1. - fluid.s_wc - fluid.s_or)
        return m_w, m_o, d_m_w, d_m_o

def pres(mesh, s, fluid, q):
    """Finite volume discretizaiton of -div(K * l(s) * grad(u)) = q.
    
    Arguments:
        mesh        ---     The mesh.
        s           ---     The saturation.
        fluid       ---     The fluid properties.
        q           ---     The source term.
    """
    # Compute K * lambda(s)
    m_w, m_o = rel_perm(s, fluid)
    m_t = m_w + m_o
    KM = np.reshape(np.hstack([m_t, m_t, m_t]).T, (3, mesh.nx, mesh.ny, mesh.nz),
                    order='F') * mesh.K
    # Compute pressure and extract fluxes
    return tpfa(mesh, KM, q)

def gen_A(mesh, v, q):
    """Matrix assembly in upwind finite-volume discretization of saturation eq."""
    nx = mesh.nx
    ny = mesh.ny
    nz = mesh.nz
    n = nx * ny * nz
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
    A = sp.spdiags(diag_vecs.T, diag_indx, n, n)
    return A

def upstream(mesh, s, fluid, v, q, T):
    """Explicit upwind finite-volum discretization of the saturation eq."""
    nx = mesh.nx
    ny = mesh.ny
    nz = mesh.nz
    n = nx * ny * nz
    # Pore volume = cell volume * porosity
    pv = (mesh.V.ravel(order='F') * mesh.por.ravel(order='F')).reshape((n, 1), order='F')
    # Inflow from wells
    fi = np.maximum(q, 0.)
    # Influx and outflux, x-faces
    XP = np.maximum(v[0], 0.)
    XN = np.minimum(v[0], 0.)
    YP = np.maximum(v[1], 0.)
    YN = np.minimum(v[1], 0.)
    ZP = np.maximum(v[2], 0.)
    ZN = np.minimum(v[2], 0.)
    # Total flux into each gridblock
    Vi = (XP[:nx, :, :] + YP[:, :ny, :] + ZP[:, :, :nz]
          - XN[1:(nx + 1), :, :] - YN[:, 1:(ny + 1), :] - ZN[:, :, 1:(nz + 1)])
    # Estimate of influx
    pm = np.min(pv / (Vi.ravel(order='F') + fi.ravel(order='F')))
    # CFL restriction
    cfl = ((1. - fluid.s_wc - fluid.s_or) / 3.) * pm
    # Number of local time steps
    n_ts = int(math.ceil(T / cfl))
    # Local time steps
    dtx = (T / n_ts) / pv
    # System matrix
    A = gen_A(mesh, v, q).tocsr()
    # A * dt / |Omega_i|
    A = sp.spdiags(dtx.T, 0, n, n).dot(A)
    # Injection
    fi = np.maximum(q, 0.) * dtx
    # Evolve in time
    for t in range(n_ts):
        print 'here', t, n_ts, cfl
        # Compute mobilities
        m_w, m_o = rel_perm(s, fluid)
        # Compute fractional flow
        f_w = m_w / (m_w + m_o)
        # Update saturation
        s += A.dot(f_w) + fi
        #c = plt.contourf(s.reshape((mesh.nx, mesh.ny, mesh.nz), order='F')[:, :, 0])
        #plt.colorbar(c)
        #plt.show()
    return s

def newt_raph(mesh, s, fluid, v, q, T):
    """Implicit scheme for the saturation equation."""
    n = mesh.nx * mesh.ny * mesh.nz
    A = gen_A(mesh, v, q)
    A = A.tocsr()
    conv = 0.
    IT = 0.
    s00 = s
    while conv == 0:
        # Time step
        dt = T / (2 ** IT)
        # Time step / pore volume
        dtx = dt / (mesh.V.ravel(order='F') * mesh.por.ravel(order='F')).reshape((n, 1), order='F')
        fi = np.maximum(q, 0) * dtx
        B = sp.spdiags(dtx.T, 0, n, n) * A
        B = B.tocsr()
        
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
                m_w, m_o, d_m_w, d_m_o = rel_perm(s, fluid, compute_der=True)
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

def test_1():
    """Simple test."""
    class Mesh:
        pass
    nx = 64
    ny = 64
    nz = 1
    mesh = Mesh()
    mesh.nx = nx
    mesh.hx = 1. / mesh.nx
    mesh.ny = ny
    mesh.hy = 1. / mesh.ny
    mesh.nz = nz
    mesh.hz = 1. / mesh.nz
    # Total number of grid blocks
    n = mesh.nx * mesh.ny * mesh.nz
    # Cell volumes
    mesh.V = np.array([mesh.hx * mesh.hy * mesh.hz])
    # Unit permeability
    mesh.K = np.ones((3, mesh.nx, mesh.ny, mesh.nz))
    x = np.linspace(0, 1, nx).reshape((nx, 1), order='F')
    X1, X2 = np.meshgrid(x, x)
    X = np.hstack([X1.reshape((nx ** 2, 1), order='F'),
                   X2.reshape((nx ** 2, 1), order='F')])
    A1 = np.ndarray((nx, nx), order='F')
    hyp1 = np.ndarray(1, order='F')
    hyp1[0] = 0.1
    A2 = np.ndarray((nx, nx), order='F')
    hyp2 = np.ndarray(1, order='F')
    hyp2[0] = 0.1
    cov = SECovarianceFunction(1)
    cov(hyp1, x, A=A1)
    cov(hyp2, x, A=A2)
    A = (A1, A2)
    f = create_DiscreteKarhunenLoeveExpansion(A, energy=.99)
    K = np.exp(2. * f(np.random.randn(f.num_input)))
    mesh.K[0, :, :, :] = K.reshape((nx, ny, nz), order='F')
    mesh.K[1, :, :, :] = mesh.K[0, :, :, :]
    mesh.K[2, :, :, :] = mesh.K[0, :, :, :]
    # Unit porosity
    mesh.por = np.ones((mesh.nx, mesh.ny, mesh.nz))
    # Production/injection
    q = np.zeros((n, 1))
    q[0] = -1.
    q[1500] = 3.
    q[-1] = -1.
    q[63] = -1.
    class Fluid:
        pass
    fluid = Fluid()
    # Viscosities
    fluid.v_w = 1.
    fluid.v_o = 1.
    # Irreducible saturations
    fluid.s_wc = 0.
    fluid.s_or = 0.
    # Initial Saturation
    s = np.zeros((n, 1))
    # Time steps
    n_t = 60
    dt = 0.7 / n_t
    c = plt.contourf(s.reshape((mesh.nx, mesh.ny, mesh.nz), order='F')[:, :, 0])
    #plt.colorbar(c)
    plt.show(block=False)
    for t in range(n_t):
        # Pressure solver
        p, v = pres(mesh, s, fluid, q)
        # Saturation solver
        #s = upstream(mesh, s, fluid, v, q, dt)
        s = newt_raph(mesh, s, fluid, v, q, dt)
        c = plt.contourf(s.reshape((mesh.nx, mesh.ny, mesh.nz))[:, :, 0])
        #c = plt.contourf(p[:, :, 0])
        #plt.colorbar(c)
        #plt.show(block=False)
        plt.draw()

if __name__ == '__main__':
    test_1()
