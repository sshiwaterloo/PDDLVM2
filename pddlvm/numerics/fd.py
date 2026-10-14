"""Stable 1D Burgers solvers (NumPy reference implementations).

Steady and transient problems are exposed as **separate** entry points so
that callers can never silently compare a steady-state model against a
time-dependent reference (a bug present in the legacy demo scripts).

Transient scheme: theta-IMEX.  Diffusion is implicit (tridiagonal solve,
unconditionally stable in the linear part); advection is explicit first-order
upwind, subject to a CFL check ``dt <= cfl * dx / max|u|``.  The legacy
forward-Euler + central-difference scheme was unconditionally unstable for
the advective term and is intentionally not reproduced here.

Both solvers accept a source term ``source(x, t)`` so they can be verified
with manufactured solutions (see pddlvm.numerics.mms).
"""

import numpy as np
from scipy.linalg import solve_banded

import tensorflow as tf


def _upwind_advection(u, dx):
    """First-order upwind discretization of u u_x at interior nodes."""
    adv = np.zeros_like(u)
    backward = u[1:-1] * (u[1:-1] - u[:-2]) / dx
    forward = u[1:-1] * (u[2:] - u[1:-1]) / dx
    adv[1:-1] = np.where(u[1:-1] >= 0.0, backward, forward)
    return adv


def _tridiag_step(rhs, a, d, c):
    """Solve (a, d, c) u_new = rhs for a tridiagonal matrix.

    ``a[j] = A[j, j-1]`` (subdiagonal, a[0] unused), ``d`` the main
    diagonal, ``c[j] = A[j, j+1]`` (superdiagonal, c[n-1] unused).
    scipy's banded layout stores ``ab[u + i - j, j] = A[i, j]``, so the
    superdiagonal row is left-shifted (leading pad) and the subdiagonal row
    is right-shifted (trailing pad) relative to the main diagonal.
    """
    n = len(d)
    ab = np.vstack([np.concatenate([[0.0], c[:-1]]),  # A[j-1, j]
                    d,
                    np.concatenate([a[1:], [0.0]])])  # A[j+1, j]
    return solve_banded((1, 1), ab, rhs)


def _as_boundary_values(bc, t):
    """Scalar or callable boundary condition -> numeric value at time t."""
    return bc(t) if callable(bc) else float(bc)


def solve_burgers_transient(x, u0, nu, source, t0, t1, dt,
                            bc_left=None, bc_right=None,
                            theta=0.5, cfl=0.9):
    """Integrate u_t + u u_x - nu u_xx = source(x, t) on the mesh x.

    Scheme: linearized theta-IMEX (semi-implicit).  Diffusion is implicit
    for the theta fraction; advection is first-order upwind with the
    implicit part linearized by freezing the upwind factor at u^n, so each
    step is a single tridiagonal solve:

        (I + theta dt (J_adv[u^n] - nu D2)) u^{n+1}
            = u^n + (1-theta) dt (nu D2 u^n - upwind(u^n u_x)) + dt S_theta

    theta = 0.5 is second-order in time (Crank-Nicolson); theta = 1.0 is
    backward Euler (first order).  Space accuracy is first order (upwind).
    The explicit advection fraction is CFL-limited: dt <= cfl dx / max|u|.

    Returns
    -------
    (u_final, t_hist, u_hist) : solution at t1 plus per-step history
        (u_hist[k] is the solution right after the k-th accepted step).
    """
    x = np.asarray(x, dtype=float)
    u = np.asarray(u0, dtype=float)
    dx = float(x[1] - x[0])
    if not np.allclose(np.diff(x), dx, rtol=1e-6):
        raise ValueError("solve_burgers_transient requires a uniform mesh")

    t = float(t0)
    t_hist = [t]
    u_hist = [u.copy()]

    while t < t1 - 1e-15:
        dt_step = min(dt, t1 - t)
        umax = max(float(np.max(np.abs(u))), 1e-12)
        dt_step = min(dt_step, cfl * dx / umax)

        n = u.shape[0]
        # explicit part
        adv = _upwind_advection(u, dx)
        d2u = np.zeros_like(u)
        d2u[1:-1] = (u[:-2] - 2.0 * u[1:-1] + u[2:]) / dx ** 2
        s_t = source(x, t)
        s_t1 = source(x, t + dt_step)
        rhs = u + dt_step * ((1.0 - theta) * (nu * d2u - adv)) \
            + dt_step * ((1.0 - theta) * s_t + theta * s_t1)

        # implicit operator: I + theta dt (J_adv - nu D2), tridiagonal
        a = np.zeros(n)  # subdiagonal a[j] = M[j, j-1]
        d = np.ones(n)   # diagonal
        c = np.zeros(n)  # superdiagonal c[j] = M[j, j+1]
        j = np.arange(1, n - 1)
        up = u[1:-1] >= 0.0
        # upwind Jacobian with the advecting factor frozen at u^n
        d[1:-1] += theta * dt_step * np.where(up, u[1:-1], -u[1:-1]) / dx
        a[1:-1] += theta * dt_step * np.where(up, -u[1:-1], 0.0) / dx
        c[1:-1] += theta * dt_step * np.where(up, 0.0, u[1:-1]) / dx
        # diffusion
        a[1:-1] += -nu * theta * dt_step / dx ** 2
        d[1:-1] += 2.0 * nu * theta * dt_step / dx ** 2
        c[1:-1] += -nu * theta * dt_step / dx ** 2

        # Dirichlet rows
        if bc_left is not None:
            a[0], d[0], c[0] = 0.0, 1.0, 0.0
            rhs[0] = _as_boundary_values(bc_left, t + dt_step)
        if bc_right is not None:
            a[-1], d[-1], c[-1] = 0.0, 1.0, 0.0
            rhs[-1] = _as_boundary_values(bc_right, t + dt_step)

        u = _tridiag_step(rhs, a, d, c)
        t += dt_step
        t_hist.append(t)
        u_hist.append(u.copy())

    return u, t_hist, u_hist


def burgers_space_time_residual(u_grid, x, t, nu, ic=None,
                                bc_left=None, bc_right=None,
                                source=None):
    """Residual of u_t + u u_x - nu u_xx = source(x, t) on a space-time grid.

    Discretization: backward Euler in time (first order), central
    differences in space (second order for the derivatives they approximate;
    the overall space accuracy is first order because the convective term
    is centered).  The returned vector concatenates

    - the interior PDE residual at (t_k, x_j), k = 1..Nt-1, j = 1..Nx-2,
    - the initial-condition mismatch u(0, x_j) - ic(x_j), all j,
    - the left/right boundary mismatches u(t_k, x_0/-1) - bc(t_k),

    so a solution of the discrete problem makes the whole vector vanish.
    With ``source=None`` the homogeneous equation is used (the demo case).

    Parameters
    ----------
    u_grid : [Nt, Nx]
    x : [Nx], t : [Nt], uniform grids
    ic : callable(x) or None
    bc_left / bc_right : scalar or callable(t) or None
    source : callable(x, t) or None

    Returns
    -------
    residual : 1D array [ (Nt-1)*(Nx-2) + Nx + 2*Nt ]
    """
    u = np.asarray(u_grid, dtype=float)
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    if not (np.allclose(np.diff(x), x[1] - x[0])
            and np.allclose(np.diff(t), t[1] - t[0])):
        raise ValueError("burgers_space_time_residual needs uniform grids")
    dx, dt = x[1] - x[0], t[1] - t[0]

    # interior points: k = 1..Nt-1, j = 1..Nx-2
    u_t = (u[1:, :] - u[:-1, :]) / dt          # [Nt-1, Nx]
    u_x = (u[:, 2:] - u[:, :-2]) / (2 * dx)    # [Nt, Nx-2]
    u_xx = (u[:, 2:] - 2 * u[:, 1:-1] + u[:, :-2]) / dx ** 2  # [Nt, Nx-2]
    adv = u[:, 1:-1] * u_x                     # [Nt, Nx-2]

    pde = u_t[:, 1:-1] + adv[1:, :] - nu * u_xx[1:, :]      # [Nt-1, Nx-2]
    if source is not None:
        tt, xx = np.meshgrid(t[1:], x[1:-1], indexing="ij")
        pde = pde - source(xx, tt)

    parts = [pde.reshape(-1)]
    if ic is not None:
        parts.append(u[0, :] - ic(x))
    if bc_left is not None:
        vals = np.array([bc_left(tk) if callable(bc_left) else bc_left
                         for tk in t])
        parts.append(u[:, 0] - vals)
    if bc_right is not None:
        vals = np.array([bc_right(tk) if callable(bc_right) else bc_right
                         for tk in t])
        parts.append(u[:, -1] - vals)
    return np.concatenate(parts)


def burgers_space_time_residual_tf(u_grid, x, t, nu, ic_fn=None,
                                   bc_left=None, bc_right=None,
                                   source=None):
    """TensorFlow counterpart of ``burgers_space_time_residual`` for use
    inside training graphs (gradients flow through ``u_grid``).

    Boundary conditions here are **scalars** (the demo uses homogeneous
    Dirichlet data); ``ic_fn`` is a vectorized callable(x) and ``source`` a
    vectorized callable(x, t).  Verified against the numpy version in
    pddlvm/tests/test_numerics.py.
    """
    x = tf.convert_to_tensor(x, u_grid.dtype)
    t = tf.convert_to_tensor(t, u_grid.dtype)
    nu = tf.convert_to_tensor(nu, u_grid.dtype)
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    u_t = (u_grid[1:, :] - u_grid[:-1, :]) / dt          # [Nt-1, Nx]
    u_x = (u_grid[:, 2:] - u_grid[:, :-2]) / (2.0 * dx)  # [Nt, Nx-2]
    u_xx = (u_grid[:, 2:] - 2.0 * u_grid[:, 1:-1]
            + u_grid[:, :-2]) / dx ** 2
    adv = u_grid[:, 1:-1] * u_x
    pde = u_t[:, 1:-1] + adv[1:, :] - nu * u_xx[1:, :]   # [Nt-1, Nx-2]
    if source is not None:
        tt, xx = tf.meshgrid(t[1:], x[1:-1], indexing="ij")
        pde = pde - source(xx, tt)

    parts = [tf.reshape(pde, [-1])]
    if ic_fn is not None:
        parts.append(u_grid[0, :] - ic_fn(x))
    if bc_left is not None:
        parts.append(u_grid[:, 0] - tf.cast(bc_left, u_grid.dtype))
    if bc_right is not None:
        parts.append(u_grid[:, -1] - tf.cast(bc_right, u_grid.dtype))
    return tf.concat(parts, axis=0)


def solve_burgers_steady(x, nu, source, bc_left, bc_right,
                         u0=None, tol=1e-12, max_iter=50):
    """Solve u u_x - nu u_xx = source(x) with damped Newton's method.

    Discretization: second-order central differences for both terms, so the
    solver is second-order accurate in dx (verified by MMS in the tests).
    The Jacobian is tridiagonal and assembled analytically; each Newton step
    is a single banded solve.  A damping halving of the update is applied
    whenever the residual norm would increase.
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    dx = float(x[1] - x[0])
    if not np.allclose(np.diff(x), dx, rtol=1e-6):
        raise ValueError("solve_burgers_steady requires a uniform mesh")
    if not (callable(source)):
        raise TypeError("source must be callable source(x) -> array")

    s = np.asarray(source(x), dtype=float)
    if u0 is None:
        u = np.linspace(float(bc_left), float(bc_right), n)
    else:
        u = np.asarray(u0, dtype=float).copy()

    def residual(uu):
        f = np.zeros_like(uu)
        f[1:-1] = (uu[1:-1] * (uu[2:] - uu[:-2]) / (2.0 * dx)
                   - nu * (uu[2:] - 2.0 * uu[1:-1] + uu[:-2]) / dx ** 2
                   - s[1:-1])
        f[0] = uu[0] - float(bc_left)
        f[-1] = uu[-1] - float(bc_right)
        return f

    def step(uu):
        # tridiagonal Jacobian: a[j] u_{j-1} + d[j] u_j + c[j] u_{j+1}
        a = np.zeros(n)
        d = np.ones(n)
        c = np.zeros(n)
        j = slice(1, -1)
        a[1:-1] = -uu[1:-1] / (2.0 * dx) - nu / dx ** 2
        c[1:-1] = uu[1:-1] / (2.0 * dx) - nu / dx ** 2
        d[1:-1] = (uu[2:] - uu[:-2]) / (2.0 * dx) + 2.0 * nu / dx ** 2
        a[0] = c[0] = 0.0
        a[-1] = c[-1] = 0.0
        return a, d, c

    for _ in range(max_iter):
        f = residual(u)
        fm = np.max(np.abs(f))
        # float64 roundoff floor: when the Newton increment J^-1 F drops
        # below eps*||u|| the update cannot change u and the residual
        # freezes at ~ ||J|| * eps * ||u|| ~ (2 nu / dx^2) * eps * ||u||.
        # Converging below that floor is numerically impossible, so the
        # acceptance threshold is the max of the requested tol and the floor.
        floor = (100.0 * np.finfo(float).eps * (2.0 * nu / dx ** 2)
                 * max(np.max(np.abs(u)), 1.0))
        if fm < max(tol, floor):
            return u
        a, d, c = step(u)
        delta = _tridiag_step(-f, a, d, c)
        alpha = 1.0
        f_new = residual(u + delta)
        for _ in range(10):
            if np.max(np.abs(f_new)) < np.max(np.abs(f)):
                break
            alpha *= 0.5
            f_new = residual(u + alpha * delta)
        u = u + alpha * delta
    raise RuntimeError(
        f"solve_burgers_steady did not converge in {max_iter} Newton steps "
        f"(last max|F| = {np.max(np.abs(f)):.3e})")
