"""2D incompressible Navier-Stokes residual with an immersed obstacle.

    u_t + (u.grad)u = -grad p + nu Lap(u) - lam u,    div(u) = 0

where ``lam`` is the Brinkman penalty field ``mask / eps`` of the obstacle --
the same body force the spectral solver (``ns2d_spectral.spectral_solve_2d``)
integrates with the exact factor ``exp(-lam dt)``.  This module provides the
PINN-style residual the PDDLVM decoder is trained to make zero, with gradients
flowing through ``u``/``v``/``p``.

Discretization matches ``ns3d.ns3d_residual_tf``: backward Euler in time
(first order), central differences in space (second order, periodic wrap).
The mean-flow pinning that drives the flow through the box is intentionally
*not* part of this residual -- the app re-pins the decoded field
(``u += U_target - mean(u)``) exactly as the spectral solver does, so the
decoder only has to learn the fluctuating field.
"""

import numpy as np
import tensorflow as tf


def _derivs2d(a, h, roll):
    """First/second derivatives along the last two axes of a [..., y, x]
    array with periodic wrap (matches ``ns3d._derivs`` restricted to 2D)."""
    da_dx = (roll(a, -1, -1) - roll(a, 1, -1)) / (2 * h)
    da_dy = (roll(a, -1, -2) - roll(a, 1, -2)) / (2 * h)
    lap = ((roll(a, -1, -1) + roll(a, 1, -1)
            + roll(a, -1, -2) + roll(a, 1, -2)
            - 4 * a) / h ** 2)
    return da_dx, da_dy, lap


def _parts(u, v, p, h, dt, nu, lam, roll, cat, reshape, ic=None,
           separate_penalty=False):
    """Common body for the numpy and TF variants (mirrors ``ns3d._ns_parts``).

    ``ic`` is an optional 2-tuple ``(u0, v0)`` of [ny, nx] arrays pinning the
    t=0 level.  ``separate_penalty`` puts the Brinkman term ``lam u`` into its
    own residual block instead of folding it into the momentum block.
    """
    u_t = (u[1:] - u[:-1]) / dt
    v_t = (v[1:] - v[:-1]) / dt
    ux, uy, ulap = _derivs2d(u, h, roll)
    vx, vy, vlap = _derivs2d(v, h, roll)
    px, py, _ = _derivs2d(p, h, roll)

    if separate_penalty:
        mom_x = u_t + (u * ux + v * uy - nu * ulap + px)[1:]
        mom_y = v_t + (u * vx + v * vy - nu * vlap + py)[1:]
        pen_x = lam * u[1:]
        pen_y = lam * v[1:]
        cont = ux + vy
        parts = [reshape(mom_x, [-1]), reshape(mom_y, [-1]),
                 reshape(pen_x, [-1]), reshape(pen_y, [-1]),
                 reshape(cont, [-1])]
    else:
        # momentum + Brinkman penalty (+ lam u) on the nt-1 interior levels
        mom_x = u_t + (u * ux + v * uy - nu * ulap + px)[1:] + lam * u[1:]
        mom_y = v_t + (u * vx + v * vy - nu * vlap + py)[1:] + lam * v[1:]
        cont = ux + vy
        parts = [reshape(mom_x, [-1]), reshape(mom_y, [-1]),
                 reshape(cont, [-1])]
    if ic is not None:
        u0, v0 = ic
        parts.append(reshape(u[0] - u0, [-1]))
        parts.append(reshape(v[0] - v0, [-1]))
    return cat(parts, 0)


def ns2d_obstacle_residual(u, v, p, x, t, nu, lam, ic=None,
                           separate_penalty=False):
    """Numpy residual of the 2D obstacle NS system.

    u, v, p : [nt, ny, nx];  x : [nx];  t : [nt];  lam : [ny, nx].
    Returns a flat vector [mom_x; mom_y; cont; (ic_u; ic_v)], or with
    ``separate_penalty`` [mom_x; mom_y; pen_x; pen_y; cont; (ic_u; ic_v)].
    """
    h = x[1] - x[0]
    dt = t[1] - t[0]
    return _parts(u, v, p, h, dt, nu, lam, np.roll, np.concatenate,
                  np.reshape, ic=ic, separate_penalty=separate_penalty)


def ns2d_obstacle_residual_tf(u, v, p, x, t, nu, lam, ic=None,
                              separate_penalty=False):
    """TensorFlow counterpart of ``ns2d_obstacle_residual`` (gradients flow
    through u/v/p).  Verified against the numpy version in
    test_ns2d_obstacle.py."""
    h = x[1] - x[0]
    dt = t[1] - t[0]
    lam = tf.cast(lam, u.dtype)
    if ic is not None:
        u0, v0 = ic
        ic = (tf.cast(u0, u.dtype), tf.cast(v0, u.dtype))
    return _parts(u, v, p, h, dt, nu, lam,
                  lambda a, s, ax: tf.roll(a, shift=s, axis=ax),
                  tf.concat, tf.reshape, ic=ic,
                  separate_penalty=separate_penalty)
