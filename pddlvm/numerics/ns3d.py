"""3D incompressible Navier-Stokes residual operators.

    u_t + (u.grad)u = -grad p + nu Lap(u) + f,   div(u) = 0

Primitive variables (u, v, w, p) with the continuity equation appended to
the residual vector (divergence penalty; PINN-style).  Two boundary modes:

- ``periodic``   : wrap-around central differences on all axes, no face
  terms (Taylor-Green regime; matches the spectral solver).
- ``dirichlet``  : central differences in the interior only, plus
  face-mismatch terms on the six boundary faces (Ethier-Steinman regime).

Discretization: backward Euler in time (first order), central differences
in space (second order), so a manufactured solution leaves an
O(dt + dx^2) residual (verified in pddlvm/tests/test_ns3d.py).
"""

import numpy as np
import tensorflow as tf


def _derivs(a, h, roll):
    """First and second derivatives along the last three axes of a [..., z, y, x]
    array with periodic wrap."""
    da_dx = (roll(a, -1, -1) - roll(a, 1, -1)) / (2 * h)
    da_dy = (roll(a, -1, -2) - roll(a, 1, -2)) / (2 * h)
    da_dz = (roll(a, -1, -3) - roll(a, 1, -3)) / (2 * h)
    lap = ((roll(a, -1, -1) + roll(a, 1, -1)
            + roll(a, -1, -2) + roll(a, 1, -2)
            + roll(a, -1, -3) + roll(a, 1, -3)
            - 6 * a) / h ** 2)
    return da_dx, da_dy, da_dz, lap


def _ns_parts(u, v, w, p, h, dt, nu, interior, roll, cat, reshape,
              bc_faces=None, ic=None):
    """Common body for the numpy and TF variants.

    ``interior`` is a slice selecting interior points (all points in the
    periodic mode); ``bc_faces`` is a list of face-mismatch tensors for the
    Dirichlet mode; ``ic`` holds the initial-condition mismatch.
    """
    u_t = (u[1:] - u[:-1]) / dt
    v_t = (v[1:] - v[:-1]) / dt
    w_t = (w[1:] - w[:-1]) / dt
    ux, uy, uz, ulap = _derivs(u, h, roll)
    vx, vy, vz, vlap = _derivs(v, h, roll)
    wx, wy, wz, wlap = _derivs(w, h, roll)
    px, py, pz, _ = _derivs(p, h, roll)

    mom_x = (u_t[:, interior, interior, interior]
             + (u * ux + v * uy + w * uz - nu * ulap
                + px)[1:, interior, interior, interior])
    mom_y = (v_t[:, interior, interior, interior]
             + (u * vx + v * vy + w * vz - nu * vlap
                + py)[1:, interior, interior, interior])
    mom_z = (w_t[:, interior, interior, interior]
             + (u * wx + v * wy + w * wz - nu * wlap
                + pz)[1:, interior, interior, interior])
    cont = (ux + vy + wz)[:, interior, interior, interior]

    parts = [reshape(mom_x, [-1]), reshape(mom_y, [-1]),
             reshape(mom_z, [-1]), reshape(cont, [-1])]
    if ic is not None:
        parts.append(reshape(ic, [-1]))
    if bc_faces:
        parts.extend(reshape(f, [-1]) for f in bc_faces)
    return cat(parts, 0)


def ns3d_residual(u, v, w, p, x, t, nu, mode="periodic", ic=None,
                  bc=None):
    """Numpy residual of the 3D NS system on a uniform cubic grid.

    u, v, w, p : [nt, n, n, n];  x : [n];  t : [nt].
    ``mode`` is ``"periodic"`` or ``"dirichlet"``.  For the Dirichlet mode,
    ``ic`` must be a callable(x, y, z) -> (u0, v0, w0) at t=0 and ``bc`` a
    callable(face_points...) — see the demo/tests; the face mismatch terms
    are appended for all six faces and all time levels.
    """
    h = x[1] - x[0]
    dt = t[1] - t[0]
    interior = slice(None) if mode == "periodic" else slice(1, -1)
    ic_parts = None
    if ic is not None:
        x4 = x[None, None, None, :]
        y4 = x[None, None, :, None]
        z4 = x[None, :, None, None]
        u0, v0, w0 = ic(x4, y4, z4)
        ic_parts = np.concatenate(
            [(u[0] - u0).reshape(-1), (v[0] - v0).reshape(-1),
             (w[0] - w0).reshape(-1)])
    faces = []
    if mode == "dirichlet":
        # six faces x three velocity components
        for comp, ref in ((u, bc[0]), (v, bc[1]), (w, bc[2])):
            for face, refv in zip(
                    (comp[:, 0], comp[:, -1], comp[:, :, 0], comp[:, :, -1],
                     comp[:, :, :, 0], comp[:, :, :, -1]),
                    (ref[:, 0], ref[:, -1], ref[:, :, 0], ref[:, :, -1],
                     ref[:, :, :, 0], ref[:, :, :, -1])):
                faces.append(face - refv)
    return _ns_parts(u, v, w, p, h, dt, nu, interior,
                     np.roll, np.concatenate, np.reshape,
                     bc_faces=faces, ic=ic_parts)


def ns3d_residual_tf(u, v, w, p, x, t, nu, mode="periodic", ic=None,
                     bc=None):
    """TensorFlow counterpart of ``ns3d_residual`` (gradients flow through
    u/v/w/p).  Verified against the numpy version in test_ns3d.py."""
    h = x[1] - x[0]
    dt = t[1] - t[0]
    interior = slice(None) if mode == "periodic" else slice(1, -1)
    ic_parts = None
    if ic is not None:
        x4 = x[None, None, None, :]
        y4 = x[None, None, :, None]
        z4 = x[None, :, None, None]
        u0, v0, w0 = ic(x4, y4, z4)
        ic_parts = tf.concat(
            [tf.reshape(u[0] - u0, [-1]), tf.reshape(v[0] - v0, [-1]),
             tf.reshape(w[0] - w0, [-1])], axis=0)
    faces = []
    if mode == "dirichlet":
        for comp, ref in ((u, bc[0]), (v, bc[1]), (w, bc[2])):
            for face, refv in zip(
                    (comp[:, 0], comp[:, -1], comp[:, :, 0], comp[:, :, -1],
                     comp[:, :, :, 0], comp[:, :, :, -1]),
                    (ref[:, 0], ref[:, -1], ref[:, :, 0], ref[:, :, -1],
                     ref[:, :, :, 0], ref[:, :, :, -1])):
                faces.append(face - refv)
    return _ns_parts(u, v, w, p, h, dt, nu, interior,
                     lambda a, s, ax: tf.roll(a, shift=s, axis=ax),
                     tf.concat, tf.reshape,
                     bc_faces=faces, ic=ic_parts)
