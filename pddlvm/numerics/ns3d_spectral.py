"""3D pseudospectral Navier-Stokes solver.

Periodic domain [0, L]^3, primitive variables, Fourier spectral
discretization with 2/3 dealiasing and exact divergence-free projection
(pressure is eliminated in Fourier space).  Time integration is classical
RK4 on the spectral ODE.

Validation (pddlvm/tests/test_ns3d.py):
- inviscid Taylor-Green vortex conserves kinetic energy (drift < 1e-4),
- viscous runs obey dE/dt = -2 nu Z to the RK4 time-discretization error.
"""

import numpy as np

# Optional FFTW backend: pyfftw (multi-threaded FFTW) is 1.1x at 96^3 but
# ~2.6x at 128^3 vs numpy's single-threaded pocketfft.  Falls back to numpy.fft
# when pyfftw / libfftw3 is unavailable.  Set PYFFTW_NUM_THREADS to tune the
# thread count (default 1 to avoid oversubscribing shared hosts).
try:
    import pyfftw
    from pyfftw.interfaces import numpy_fft as _fft_backend
    pyfftw.interfaces.cache.enable()
    import os as _os
    pyfftw.config.NUM_THREADS = int(_os.environ.get("PYFFTW_NUM_THREADS", "1"))
    _HAVE_FFTW = True
except ImportError:  # pragma: no cover - exercised only without pyfftw
    import numpy.fft as _fft_backend
    _HAVE_FFTW = False


def _fftn(a):
    return _fft_backend.fftn(a)


def _ifftn(a):
    return _fft_backend.ifftn(a)



def _wave_numbers(n, L):
    k = np.fft.fftfreq(n, d=L / n) * 2.0 * np.pi
    kx = k[None, None, :]
    ky = k[None, :, None]
    kz = k[:, None, None]
    return kx, ky, kz


def _project(fu, fv, fw, kx, ky, kz):
    """Project the Fourier coefficients onto the divergence-free subspace
    (this absorbs the pressure gradient exactly)."""
    k2 = kx ** 2 + ky ** 2 + kz ** 2
    k2 = np.where(k2 == 0, 1.0, k2)
    div = kx * fu + ky * fv + kz * fw
    fu = fu - kx * div / k2
    fv = fv - ky * div / k2
    fw = fw - kz * div / k2
    return fu, fv, fw


def _dealias_mask(n):
    kmax = n // 2
    cutoff = int(2.0 / 3.0 * kmax)
    m = np.ones((n, n, n), dtype=bool)
    kf = np.fft.fftfreq(n)
    kx = kf[None, None, :]
    ky = kf[None, :, None]
    kz = kf[:, None, None]
    krad = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)
    m[krad > cutoff] = False
    return m


def q_criterion_3d(u, v, w, L=2.0 * np.pi):
    """Q-criterion scalar field (vortex identification) on a cubic periodic
    box: ``Q = 0.5 * (|omega|^2 - |grad u|^2)``, positive inside vortex cores
    where rotation dominates strain.  ``u, v, w`` are (n, n, n) velocity
    fields; returns the (n, n, n) Q field."""
    n = u.shape[0]
    kx, ky, kz = _wave_numbers(n, L)
    fu = _fftn(u); fv = _fftn(v); fw = _fftn(w)
    ux = np.real(_ifftn(1j * kx * fu))
    uy = np.real(_ifftn(1j * ky * fu))
    uz = np.real(_ifftn(1j * kz * fu))
    vx = np.real(_ifftn(1j * kx * fv))
    vy = np.real(_ifftn(1j * ky * fv))
    vz = np.real(_ifftn(1j * kz * fv))
    wx = np.real(_ifftn(1j * kx * fw))
    wy = np.real(_ifftn(1j * ky * fw))
    wz = np.real(_ifftn(1j * kz * fw))
    ox = wy - vz
    oy = uz - wx
    oz = vx - uy
    vor2 = ox ** 2 + oy ** 2 + oz ** 2
    grad2 = (ux ** 2 + uy ** 2 + uz ** 2 + vx ** 2 + vy ** 2 + vz ** 2
             + wx ** 2 + wy ** 2 + wz ** 2)
    return 0.5 * (vor2 - grad2)


def spectral_solve(u0, v0, w0, nu, dt, n_steps, L=2.0 * np.pi,
                   snapshot_every=None):
    """Integrate the incompressible NS equations with RK4.

    Returns
    -------
    (u, v, w, snapshots) : final fields plus a list of (step, (u,v,w))
        snapshots recorded every ``snapshot_every`` steps (including step 0).
    """
    n = u0.shape[0]
    kx, ky, kz = _wave_numbers(n, L)
    k2 = kx ** 2 + ky ** 2 + kz ** 2
    mask = _dealias_mask(n)

    fu = _fftn(u0)
    fv = _fftn(v0)
    fw = _fftn(w0)
    fu, fv, fw = _project(fu, fv, fw, kx, ky, kz)

    snapshots = []
    if snapshot_every is not None:
        u = np.real(_ifftn(fu))
        v = np.real(_ifftn(fv))
        w = np.real(_ifftn(fw))
        snapshots.append((0, (u.copy(), v.copy(), w.copy())))

    def rhs(fu, fv, fw):
        # nonlinear term in physical space (dealiased)
        u = np.real(_ifftn(fu * mask))
        v = np.real(_ifftn(fv * mask))
        w = np.real(_ifftn(fw * mask))
        ux = np.real(_ifftn(1j * kx * fu * mask))
        uy = np.real(_ifftn(1j * ky * fu * mask))
        uz = np.real(_ifftn(1j * kz * fu * mask))
        vx = np.real(_ifftn(1j * kx * fv * mask))
        vy = np.real(_ifftn(1j * ky * fv * mask))
        vz = np.real(_ifftn(1j * kz * fv * mask))
        wx = np.real(_ifftn(1j * kx * fw * mask))
        wy = np.real(_ifftn(1j * ky * fw * mask))
        wz = np.real(_ifftn(1j * kz * fw * mask))

        nx = -(u * ux + v * uy + w * uz)
        ny = -(u * vx + v * vy + w * vz)
        nz = -(u * wx + v * wy + w * wz)
        gx, gy, gz = _project(_fftn(nx), _fftn(ny),
                              _fftn(nz), kx, ky, kz)
        return (gx - nu * k2 * fu,
                gy - nu * k2 * fv,
                gz - nu * k2 * fw)

    for step in range(n_steps):
        k1x, k1y, k1z = rhs(fu, fv, fw)
        k2x, k2y, k2z = rhs(fu + 0.5 * dt * k1x, fv + 0.5 * dt * k1y,
                            fw + 0.5 * dt * k1z)
        k3x, k3y, k3z = rhs(fu + 0.5 * dt * k2x, fv + 0.5 * dt * k2y,
                            fw + 0.5 * dt * k2z)
        k4x, k4y, k4z = rhs(fu + dt * k3x, fv + dt * k3y, fw + dt * k3z)
        fu = fu + dt / 6.0 * (k1x + 2 * k2x + 2 * k3x + k4x)
        fv = fv + dt / 6.0 * (k1y + 2 * k2y + 2 * k3y + k4y)
        fw = fw + dt / 6.0 * (k1z + 2 * k2z + 2 * k3z + k4z)
        fu, fv, fw = _project(fu, fv, fw, kx, ky, kz)

        if snapshot_every is not None and (step + 1) % snapshot_every == 0:
            u = np.real(_ifftn(fu))
            v = np.real(_ifftn(fv))
            w = np.real(_ifftn(fw))
            snapshots.append((step + 1, (u.copy(), v.copy(), w.copy())))

    u = np.real(_ifftn(fu))
    v = np.real(_ifftn(fv))
    w = np.real(_ifftn(fw))
    return u, v, w, snapshots


def spectral_solve_obstacle(u0, v0, w0, nu, dt, n_steps, L=2.0 * np.pi,
                            U_target=None, force_x=None, penalty=None,
                            diameter=1.0, hyper_nu=0.0,
                            frontal_area=None,
                            snapshot_every=None, probe_every=None):
    """3D NS with immersed solid obstacles (sphere / pillar).

    Same design as the hardened 2D obstacle solver (``ns2d_spectral``):
    Lawson-type RK4 with **exact exponential integrating factors** for the
    viscous/hyperviscous terms (Fourier space) and the Brinkman penalty
    (physical space), so ``dt`` is set by the advective CFL only -- never by
    ``nu*k^2`` or the penalty stiffness.  Mean-flow pinning ``U_target``
    (Re = U D / nu exact) or constant body force ``force_x``; IBM drag/lift
    read off the penalty: ``F = int lambda*u dV``, ``Cd = 2 Fx / (U^2 A)``
    with reference area ``A`` = ``frontal_area`` if given, else the sphere
    frontal area ``pi D^2 / 4`` (so ``Cd = 8 Fx / (pi D^2 U^2)``).

    Returns (u, v, w, snapshots, probes); probes are (step, Cd, Cl_y, Cl_z)
    (``None`` when ``probe_every`` is not requested).
    """
    n = u0.shape[0]
    kx, ky, kz = _wave_numbers(n, L)
    k2 = kx ** 2 + ky ** 2 + kz ** 2
    dmask = _dealias_mask(n)
    lam = penalty if penalty is not None else np.zeros_like(u0)
    vol = L ** 3
    u_ref2 = (U_target ** 2 if U_target is not None else 1.0)
    area = frontal_area if frontal_area is not None else (np.pi * diameter ** 2 / 4.0)
    denom = 0.5 * area * u_ref2

    damp = nu * k2 + hyper_nu * k2 ** 2
    e_v_half = np.exp(-0.5 * dt * damp)
    e_v_full = np.exp(-dt * damp)
    e_p_half = np.exp(-0.5 * dt * lam)
    e_p_full = np.exp(-dt * lam)

    fu = _fftn(u0)
    fv = _fftn(v0)
    fw = _fftn(w0)
    fu, fv, fw = _project(fu, fv, fw, kx, ky, kz)

    snapshots = []
    if snapshot_every is not None:
        u = np.real(_ifftn(fu))
        v = np.real(_ifftn(fv))
        w = np.real(_ifftn(fw))
        snapshots.append((0, (u.copy(), v.copy(), w.copy())))
    probes = [] if probe_every is not None else None

    def rhs(fu, fv, fw):
        u = np.real(_ifftn(fu * dmask))
        v = np.real(_ifftn(fv * dmask))
        w = np.real(_ifftn(fw * dmask))
        ux = np.real(_ifftn(1j * kx * fu * dmask))
        uy = np.real(_ifftn(1j * ky * fu * dmask))
        uz = np.real(_ifftn(1j * kz * fu * dmask))
        vx = np.real(_ifftn(1j * kx * fv * dmask))
        vy = np.real(_ifftn(1j * ky * fv * dmask))
        vz = np.real(_ifftn(1j * kz * fv * dmask))
        wx = np.real(_ifftn(1j * kx * fw * dmask))
        wy = np.real(_ifftn(1j * ky * fw * dmask))
        wz = np.real(_ifftn(1j * kz * fw * dmask))
        nx = -(u * ux + v * uy + w * uz)
        ny = -(u * vx + v * vy + w * vz)
        nz = -(u * wx + v * wy + w * wz)
        return _project(_fftn(nx), _fftn(ny), _fftn(nz),
                        kx, ky, kz)

    def lawson(a, field_u, field_v, field_w, rx=None, ry=None, rz=None):
        ev = e_v_half if a == 0.5 else e_v_full
        ep = e_p_half if a == 0.5 else e_p_full
        u = field_u + (a * dt * rx if rx is not None else 0.0)
        v = field_v + (a * dt * ry if ry is not None else 0.0)
        w = field_w + (a * dt * rz if rz is not None else 0.0)
        u = np.real(_ifftn(ev * _fftn(u))) * ep
        v = np.real(_ifftn(ev * _fftn(v))) * ep
        w = np.real(_ifftn(ev * _fftn(w))) * ep
        return u, v, w

    def record(step):
        if probe_every is not None and (step + 1) % probe_every == 0:
            u = np.real(_ifftn(fu * dmask))
            v = np.real(_ifftn(fv * dmask))
            w = np.real(_ifftn(fw * dmask))
            fx = np.sum(lam * u) * (vol / (n ** 3))
            fy = np.sum(lam * v) * (vol / (n ** 3))
            fz = np.sum(lam * w) * (vol / (n ** 3))
            probes.append((step + 1, 2.0 * fx / denom, 2.0 * fy / denom,
                           2.0 * fz / denom))
        if snapshot_every is not None and (step + 1) % snapshot_every == 0:
            u = np.real(_ifftn(fu))
            v = np.real(_ifftn(fv))
            w = np.real(_ifftn(fw))
            snapshots.append((step + 1, (u.copy(), v.copy(), w.copy())))

    for step in range(n_steps):
        u = np.real(_ifftn(fu))
        v = np.real(_ifftn(fv))
        w = np.real(_ifftn(fw))
        gx, gy, gz = rhs(fu, fv, fw)
        k1x, k1y, k1z = np.real(_ifftn(gx)), np.real(_ifftn(gy)), \
            np.real(_ifftn(gz))
        u1, v1, w1 = lawson(0.5, u, v, w, k1x, k1y, k1z)
        fu1, fv1, fw1 = _project(_fftn(u1), _fftn(v1),
                                 _fftn(w1), kx, ky, kz)
        gx, gy, gz = rhs(fu1, fv1, fw1)
        k2x, k2y, k2z = np.real(_ifftn(gx)), np.real(_ifftn(gy)), \
            np.real(_ifftn(gz))
        u2, v2, w2 = lawson(0.5, u, v, w, k2x, k2y, k2z)
        fu2, fv2, fw2 = _project(_fftn(u2), _fftn(v2),
                                 _fftn(w2), kx, ky, kz)
        gx, gy, gz = rhs(fu2, fv2, fw2)
        k3x, k3y, k3z = np.real(_ifftn(gx)), np.real(_ifftn(gy)), \
            np.real(_ifftn(gz))
        u3, v3, w3 = lawson(1.0, u, v, w, k3x, k3y, k3z)
        fu3, fv3, fw3 = _project(_fftn(u3), _fftn(v3),
                                 _fftn(w3), kx, ky, kz)
        gx, gy, gz = rhs(fu3, fv3, fw3)
        k4x, k4y, k4z = np.real(_ifftn(gx)), np.real(_ifftn(gy)), \
            np.real(_ifftn(gz))
        u_new, v_new, w_new = lawson(
            1.0, u, v, w,
            (k1x + 2.0 * k2x + 2.0 * k3x + k4x) / 6.0,
            (k1y + 2.0 * k2y + 2.0 * k3y + k4y) / 6.0,
            (k1z + 2.0 * k2z + 2.0 * k3z + k4z) / 6.0)
        if U_target is not None:
            u_new += U_target - u_new.mean()
            v_new -= v_new.mean()
            w_new -= w_new.mean()
        elif force_x is not None:
            u_new += dt * force_x
        fu, fv, fw = _project(_fftn(u_new), _fftn(v_new),
                              _fftn(w_new), kx, ky, kz)
        record(step)

    u = np.real(_ifftn(fu))
    v = np.real(_ifftn(fv))
    w = np.real(_ifftn(fw))
    return u, v, w, snapshots, probes
