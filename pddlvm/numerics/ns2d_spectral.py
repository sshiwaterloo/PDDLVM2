"""2D pseudospectral Navier-Stokes solver with immersed solid obstacles.

Periodic domain [0, Lx] x [0, Ly], primitive variables, Fourier spectral
discretization with 2/3 dealiasing and exact divergence-free projection
(pressure eliminated in Fourier space).  Time integration is Lawson-type RK4:
the viscous term and the Brinkman penalty are integrated with **exact
exponential integrating factors** (diagonal in Fourier / physical space), so
the time step is set by the advective CFL only -- never by nu*k^2 or the
penalty stiffness (validation: viscous Taylor-Green decays at the analytic
rate to machine precision).

Flow driving: either pinning ``U_target`` (fixes the zero-wavenumber mode
of ``u``, so ``Re = U_target * D / nu`` is exact) or a constant body force
``force_x`` (mean velocity self-adjusts; the feedback-free alternative).
The force the obstacle exerts is read off the penalty (IBM formula):
``F = int lambda*u dV``, giving drag/lift coefficients.

Validation (pddlvm/tests/test_obstacles.py):
- inviscid Taylor-Green vortex conserves kinetic energy,
- pure penalty decay matches the analytic exp(-lambda t) solution,
- solid region velocity is driven to ~0 by the penalty.
"""

import numpy as np


def _wave_numbers(nx, ny, Lx, Ly):
    kx = np.fft.fftfreq(nx, d=Lx / nx) * 2.0 * np.pi
    ky = np.fft.fftfreq(ny, d=Ly / ny) * 2.0 * np.pi
    return kx[None, :], ky[:, None]


def _project(fu, fv, kx, ky):
    """Project onto the divergence-free subspace (absorbs pressure exactly)."""
    k2 = kx ** 2 + ky ** 2
    k2 = np.where(k2 == 0, 1.0, k2)
    div = kx * fu + ky * fv
    return fu - kx * div / k2, fv - ky * div / k2


def _dealias_mask(nx, ny):
    kmax = min(nx, ny) // 2
    cutoff = int(2.0 / 3.0 * kmax)
    kxf = np.fft.fftfreq(nx)
    kyf = np.fft.fftfreq(ny)
    krad = np.sqrt(kxf[None, :] ** 2 + kyf[:, None] ** 2)
    return krad <= cutoff


def spectral_solve_2d(u0, v0, nu, dt, n_steps, Lx=8.0, Ly=8.0,
                      U_target=None, force_x=None, penalty=None,
                      diameter=1.0, hyper_nu=0.0,
                      snapshot_every=None, probe_every=None):
    """Integrate the incompressible 2D NS equations with RK4.

    Parameters
    ----------
    u0, v0 : (ny, nx) float arrays, initial velocity field.
    nu : kinematic viscosity.
    U_target : optional float; pin the mean of ``u`` to this speed (flow
        driven through the obstacle).  None => free decay (validation runs).
    force_x : optional constant body force driving the flow (alternative to
        pinning; mean velocity emerges from the momentum balance).
    penalty : optional (ny, nx) Brinkman field ``lambda(x) = mask/eps``;
        None => no obstacle.
    diameter : obstacle diameter, for Cd/Cl nondimensionalization.
    hyper_nu : optional hyperviscosity coefficient; adds -hyper_nu * k^4 to
        the viscous exponential factor (damps grid-scale shear-layer modes
        that periodic recycling would otherwise re-amplify).
    snapshot_every, probe_every : optional sampling intervals.  Probes record
        (step, Cd, Cl) with the IBM drag/lift formulas.

    Returns
    -------
    u, v, snapshots, probes : final fields; list of (step, (u, v)) snapshots;
        list of (step, Cd, Cl) probe readings (None if not requested).
    """
    ny, nx = u0.shape
    kx, ky = _wave_numbers(nx, ny, Lx, Ly)
    k2 = kx ** 2 + ky ** 2
    dmask = _dealias_mask(nx, ny)
    lam = penalty if penalty is not None else np.zeros_like(u0)
    vol = Lx * Ly
    denom = 0.5 * diameter * (U_target ** 2 if U_target is not None else 1.0)

    # exact exponential integrating factors: viscous + hyperviscous
    # (Fourier), penalty (physical)
    damp = nu * k2 + hyper_nu * k2 ** 2
    e_v_half = np.exp(-0.5 * dt * damp)
    e_v_full = np.exp(-dt * damp)
    e_p_half = np.exp(-0.5 * dt * lam)
    e_p_full = np.exp(-dt * lam)

    fu = np.fft.fftn(u0)
    fv = np.fft.fftn(v0)
    fu, fv = _project(fu, fv, kx, ky)

    snapshots = []
    if snapshot_every is not None:
        u = np.real(np.fft.ifftn(fu))
        v = np.real(np.fft.ifftn(fv))
        snapshots.append((0, (u.copy(), v.copy())))

    probes = None
    if probe_every is not None:
        probes = []

    def rhs(fu, fv):
        # nonlinear term only (dealiased); viscous + penalty are implicit
        u = np.real(np.fft.ifftn(fu * dmask))
        v = np.real(np.fft.ifftn(fv * dmask))
        ux = np.real(np.fft.ifftn(1j * kx * fu * dmask))
        uy = np.real(np.fft.ifftn(1j * ky * fu * dmask))
        vx = np.real(np.fft.ifftn(1j * kx * fv * dmask))
        vy = np.real(np.fft.ifftn(1j * ky * fv * dmask))
        nx_ = -(u * ux + v * uy)
        ny_ = -(u * vx + v * vy)
        return _project(np.fft.fftn(nx_), np.fft.fftn(ny_), kx, ky)

    def lawson(a, field_u, field_v, with_rhs_u=None, with_rhs_v=None):
        """Apply E(a) = exp(-a dt (nu k^2 + lambda)) to a field (or to the
        combination field + a dt RHS, Lawson structure).  Viscous factor in
        Fourier space, penalty factor pointwise (split product, O(dt^2)
        coupling error confined to the mask transition zone)."""
        ev = e_v_half if a == 0.5 else e_v_full
        ep = e_p_half if a == 0.5 else e_p_full
        u = field_u + (a * dt * with_rhs_u if with_rhs_u is not None else 0.0)
        v = field_v + (a * dt * with_rhs_v if with_rhs_v is not None else 0.0)
        u = np.real(np.fft.ifftn(ev * np.fft.fftn(u))) * ep
        v = np.real(np.fft.ifftn(ev * np.fft.fftn(v))) * ep
        return u, v

    def record(step):
        if probe_every is not None and (step + 1) % probe_every == 0:
            u = np.real(np.fft.ifftn(fu * dmask))
            v = np.real(np.fft.ifftn(fv * dmask))
            fx = np.sum(lam * u) * (Lx * Ly / (nx * ny))
            fy = np.sum(lam * v) * (Lx * Ly / (nx * ny))
            probes.append((step + 1, 2.0 * fx / denom, 2.0 * fy / denom))
        if snapshot_every is not None and (step + 1) % snapshot_every == 0:
            u = np.real(np.fft.ifftn(fu))
            v = np.real(np.fft.ifftn(fv))
            snapshots.append((step + 1, (u.copy(), v.copy())))

    for step in range(n_steps):
        u = np.real(np.fft.ifftn(fu))
        v = np.real(np.fft.ifftn(fv))
        gx, gy = rhs(fu, fv)
        k1x = np.real(np.fft.ifftn(gx))
        k1y = np.real(np.fft.ifftn(gy))
        u1, v1 = lawson(0.5, u, v, k1x, k1y)
        fu1, fv1 = _project(np.fft.fftn(u1), np.fft.fftn(v1), kx, ky)
        gx, gy = rhs(fu1, fv1)
        k2x = np.real(np.fft.ifftn(gx))
        k2y = np.real(np.fft.ifftn(gy))
        u2, v2 = lawson(0.5, u, v, k2x, k2y)
        fu2, fv2 = _project(np.fft.fftn(u2), np.fft.fftn(v2), kx, ky)
        gx, gy = rhs(fu2, fv2)
        k3x = np.real(np.fft.ifftn(gx))
        k3y = np.real(np.fft.ifftn(gy))
        u3, v3 = lawson(1.0, u, v, k3x, k3y)
        fu3, fv3 = _project(np.fft.fftn(u3), np.fft.fftn(v3), kx, ky)
        gx, gy = rhs(fu3, fv3)
        k4x = np.real(np.fft.ifftn(gx))
        k4y = np.real(np.fft.ifftn(gy))
        u_new, v_new = lawson(
            1.0, u, v,
            (k1x + 2.0 * k2x + 2.0 * k3x + k4x) / 6.0,
            (k1y + 2.0 * k2y + 2.0 * k3y + k4y) / 6.0)
        if U_target is not None:
            u_new += U_target - u_new.mean()
            v_new -= v_new.mean()
        elif force_x is not None:
            u_new += dt * force_x
        fu, fv = _project(np.fft.fftn(u_new), np.fft.fftn(v_new), kx, ky)

        record(step)

    u = np.real(np.fft.ifftn(fu))
    v = np.real(np.fft.ifftn(fv))
    return u, v, snapshots, probes


def vorticity(u, v, Lx=8.0, Ly=8.0):
    """Vorticity ``w = dv/dx - du/dy`` from Fourier derivatives (shape ny, nx)."""
    ny, nx = u.shape
    kx, ky = _wave_numbers(nx, ny, Lx, Ly)
    fu = np.fft.fftn(u)
    fv = np.fft.fftn(v)
    return np.real(np.fft.ifftn(1j * kx * fv - 1j * ky * fu))
