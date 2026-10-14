"""3D heat equation operators for the time-dependent PDDLVM demo.

The PDE is u_t = nu Lap(u) + f on [0,1]^3 with homogeneous Dirichlet
boundary conditions.  Discretization: backward Euler in time (first order)
and the standard 7-point stencil for the Laplacian (second order), so the
space-time residual of a manufactured solution is O(dt + dx^2) (verified
in pddlvm/tests/test_heat3d.py).

Ground truth: the manufactured solution (pddlvm.numerics.mms.heat3d_mms_u)
is analytic, so the reference used for comparison is independent of both
the FD operator and the network.
"""

import numpy as np
import tensorflow as tf


def _laplacian_7pt_np(u, dx):
    out = np.zeros_like(u)
    out[:, 1:-1, 1:-1, 1:-1] = (
        u[:, 1:-1, 1:-1, 2:] + u[:, 1:-1, 1:-1, :-2]
        + u[:, 1:-1, 2:, 1:-1] + u[:, 1:-1, :-2, 1:-1]
        + u[:, 2:, 1:-1, 1:-1] + u[:, :-2, 1:-1, 1:-1]
        - 6.0 * u[:, 1:-1, 1:-1, 1:-1]) / dx ** 2
    return out


def heat3d_residual(u_grid, x, y, z, t, nu, ic_fn=None, source_fn=None):
    """Residual of u_t - nu Lap(u) = f on a space-time grid.

    ``u_grid`` has shape [Nt, Nz, Ny, Nx] (first axis time).  Returns the
    concatenation of: interior PDE residuals (time levels 1..Nt-1, interior
    spatial points), the initial-condition mismatch, and the homogeneous
    boundary mismatches on the six faces.
    """
    u = np.asarray(u_grid, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    t = np.asarray(t, dtype=float)
    dx = x[1] - x[0]
    if not (np.allclose(np.diff(x), dx) and np.allclose(np.diff(y), dx)
            and np.allclose(np.diff(z), dx)):
        raise ValueError("heat3d_residual needs uniform cubic grids")
    dt = t[1] - t[0]

    u_t = (u[1:] - u[:-1]) / dt            # [Nt-1, Nz, Ny, Nx]
    lap = _laplacian_7pt_np(u, dx)         # [Nt, Nz, Ny, Nx]
    pde = (u_t[:, 1:-1, 1:-1, 1:-1]
           - nu * lap[1:, 1:-1, 1:-1, 1:-1])     # [Nt-1, Nz-2, Ny-2, Nx-2]
    if source_fn is not None:
        xg = x[1:-1][None, None, None, :]
        yg = y[1:-1][None, None, :, None]
        zg = z[1:-1][None, :, None, None]
        tg = t[1:][:, None, None, None]
        pde = pde - source_fn(xg, yg, zg, tg)

    parts = [pde.reshape(-1)]
    if ic_fn is not None:
        xg = x[None, None, None, :]
        yg = y[None, None, :, None]
        zg = z[None, :, None, None]
        parts.append((u[0] - ic_fn(xg, yg, zg, 0.0)).reshape(-1))
    # homogeneous Dirichlet faces
    for face in (u[:, 0], u[:, -1], u[:, :, 0], u[:, :, -1],
                 u[:, :, :, 0], u[:, :, :, -1]):
        parts.append(face.reshape(-1))
    return np.concatenate(parts)


def heat3d_residual_tf(u_grid, x, y, z, t, nu, ic_fn=None, source_fn=None):
    """TensorFlow counterpart of ``heat3d_residual`` (gradients flow
    through ``u_grid``).  Verified against the numpy version in
    pddlvm/tests/test_heat3d.py.
    """
    x = tf.convert_to_tensor(x, u_grid.dtype)
    y = tf.convert_to_tensor(y, u_grid.dtype)
    z = tf.convert_to_tensor(z, u_grid.dtype)
    t = tf.convert_to_tensor(t, u_grid.dtype)
    nu = tf.convert_to_tensor(nu, u_grid.dtype)
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    u_t = (u_grid[1:] - u_grid[:-1]) / dt
    interior = (
        u_grid[:, 1:-1, 1:-1, 2:] + u_grid[:, 1:-1, 1:-1, :-2]
        + u_grid[:, 1:-1, 2:, 1:-1] + u_grid[:, 1:-1, :-2, 1:-1]
        + u_grid[:, 2:, 1:-1, 1:-1] + u_grid[:, :-2, 1:-1, 1:-1]
        - 6.0 * u_grid[:, 1:-1, 1:-1, 1:-1]) / dx ** 2
    pde = u_t[:, 1:-1, 1:-1, 1:-1] - nu * interior[1:]
    if source_fn is not None:
        xg = x[1:-1][None, None, None, :]
        yg = y[1:-1][None, None, :, None]
        zg = z[1:-1][None, :, None, None]
        tg = t[1:][:, None, None, None]
        pde = pde - source_fn(xg, yg, zg, tg)

    parts = [tf.reshape(pde, [-1])]
    if ic_fn is not None:
        xg = x[None, None, None, :]
        yg = y[None, None, :, None]
        zg = z[None, :, None, None]
        parts.append(tf.reshape(
            u_grid[0] - ic_fn(xg, yg, zg, 0.0), [-1]))
    for face in (u_grid[:, 0], u_grid[:, -1], u_grid[:, :, 0],
                 u_grid[:, :, -1], u_grid[:, :, :, 0], u_grid[:, :, :, -1]):
        parts.append(tf.reshape(face, [-1]))
    return tf.concat(parts, axis=0)
