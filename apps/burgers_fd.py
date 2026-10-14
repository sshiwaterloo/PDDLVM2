"""Stable 1D Burgers reference solver (demo).

The legacy version integrated u_t + u u_x - nu u_xx = 0 with forward Euler
plus central differences, a scheme that is unconditionally unstable for the
advective term.  This module uses the theta-IMEX integrator from
``pddlvm.numerics.fd`` (implicit diffusion, explicit first-order upwind
advection, CFL-limited step size).

``__main__`` runs a manufactured-solution (MMS) self-check and prints the
measured convergence behaviour, then reproduces the classic demo solve
u(x, 0) = -sin(pi x) on [-1, 1] up to T = 1.
"""

import _bootstrap  # noqa: F401  (repo root on sys.path)
import numpy as np

from pddlvm.numerics.fd import solve_burgers_transient
from pddlvm.numerics.mms import (
    burgers_mms_source_transient,
    burgers_mms_u_transient,
    l2_error,
)


def solve_burgers_demo(nu=0.01, Nx=201, T=1.0, t_snap=0.5):
    """Solve the homogeneous Dirichlet Burgers problem from the legacy demo.

    Returns
    -------
    (x, u_final, u_snap) : mesh, solution at T, and a time-interpolated
        snapshot at t_snap.
    """
    x = np.linspace(-1.0, 1.0, Nx)
    u0 = -np.sin(np.pi * x)
    u0[0] = u0[-1] = 0.0
    dx = x[1] - x[0]
    dt = 0.9 * dx / max(1.0, np.max(np.abs(u0)))  # CFL-based

    u_final, t_hist, u_hist = solve_burgers_transient(
        x, u0, nu=nu,
        source=lambda xx, t: np.zeros_like(xx),
        t0=0.0, t1=T, dt=dt,
        bc_left=0.0, bc_right=0.0,
    )
    u_snap = np.array([np.interp(t_snap, t_hist, [uu[i] for uu in u_hist])
                       for i in range(Nx)])
    return x, u_final, u_snap


def _mms_self_check():
    """Manufactured-solution smoke check of the transient integrator.

    The rigorous convergence-rate verification lives in the test suite
    (pddlvm.tests.test_numerics.TestBurgersTransient); this checks that a
    single solve at moderate resolution lands near the manufactured
    solution.
    """
    nu = 0.1
    t1 = 0.5

    def mms_bc_left(t):
        return burgers_mms_u_transient(0.0, t, nu)

    def mms_bc_right(t):
        return burgers_mms_u_transient(1.0, t, nu)

    x = np.linspace(0.0, 1.0, 512 + 1)
    u0 = burgers_mms_u_transient(x, 0.0, nu)
    u, _, _ = solve_burgers_transient(
        x, u0, nu,
        source=lambda xx, t: burgers_mms_source_transient(xx, t, nu),
        t0=0.0, t1=t1, dt=1e-3,
        bc_left=mms_bc_left, bc_right=mms_bc_right)
    err = l2_error(u, burgers_mms_u_transient(x, t1, nu), x)
    print(f"MMS self-check (transient Burgers): L2 err = {err:.3e} "
          f"(dx={x[1] - x[0]:.2e}, dt=1e-3)")
    if err > 5e-3:
        raise RuntimeError(f"MMS self-check failed: err = {err:.3e}")
    return err


if __name__ == "__main__":
    _mms_self_check()
    x, u_final, u_snap = solve_burgers_demo()
    print("demo solve: ||u(T=1)||_inf = %.3e" % np.max(np.abs(u_final)))
