"""MMS verification of the 3D heat equation operators.

The manufactured solution (two sine modes with exponential time decay,
homogeneous Dirichlet data) lives in pddlvm.numerics.mms.  The space-time
residual of the FD operator applied to it is O(dt + dx^2); the test checks
the order and the magnitude, plus numpy/TensorFlow parity and the exact
zero solution.
"""

import unittest

import numpy as np
import tensorflow as tf

from pddlvm.numerics.heat3d import heat3d_residual, heat3d_residual_tf
from pddlvm.numerics.mms import (
    heat3d_mms_source,
    heat3d_mms_u,
)


def _grids(n, nt):
    x = np.linspace(0.0, 1.0, n)
    t = np.linspace(0.0, 0.5, nt)
    # u_grid [Nt, Nz, Ny, Nx]
    xg = x[None, None, None, :]
    yg = x[None, None, :, None]
    zg = x[None, :, None, None]
    tg = t[:, None, None, None]
    u = heat3d_mms_u(xg, yg, zg, tg)
    return x, t, u


class TestHeat3dResidual(unittest.TestCase):
    def test_exact_zero_solution(self):
        u = np.zeros((5, 4, 4, 4))
        x = np.linspace(0.0, 1.0, 4)
        t = np.linspace(0.0, 0.4, 5)
        r = heat3d_residual(u, x, x, x, t, nu=0.05)
        self.assertEqual(np.max(np.abs(r)), 0.0)
        r_tf = heat3d_residual_tf(tf.constant(u, tf.float64),
                                  tf.constant(x, tf.float64),
                                  tf.constant(x, tf.float64),
                                  tf.constant(x, tf.float64),
                                  tf.constant(t, tf.float64), 0.05).numpy()
        self.assertEqual(np.max(np.abs(r_tf)), 0.0)

    def test_mms_truncation_scaling(self):
        x_c, t_c, u_c = _grids(9, 9)
        x_f, t_f, u_f = _grids(17, 17)

        def residual(x, t, u):
            xg = x[None, None, None, :]
            yg = x[None, None, :, None]
            zg = x[None, :, None, None]
            r = heat3d_residual(
                u, x, x, x, t, nu=0.05,
                ic_fn=lambda xv, yv, zv, tv: heat3d_mms_u(
                    xv, yv, zv, np.zeros_like(tv)),
                source_fn=heat3d_mms_source)
            return np.max(np.abs(r))

        r_coarse = residual(x_c, t_c, u_c)
        r_fine = residual(x_f, t_f, u_f)
        print(f"heat3d residual MMS: coarse {r_coarse:.2e}, "
              f"fine {r_fine:.2e}")
        # truncation is O(dt + dx^2); with dt ~ dx the ratio is ~4
        self.assertGreater(r_coarse / r_fine, 2.5)
        self.assertLess(r_coarse / r_fine, 4.5)
        self.assertLess(r_fine, 3.5)

    def test_tf_matches_numpy(self):
        rng = np.random.default_rng(5)
        x = np.linspace(0.0, 1.0, 7)
        t = np.linspace(0.0, 0.4, 5)
        u = rng.normal(size=(5, 7, 7, 7))

        def ic(xv, yv, zv, tv):
            return np.sin(np.pi * xv) * np.sin(np.pi * yv) * np.sin(
                np.pi * zv)

        r_np = heat3d_residual(u, x, x, x, t, 0.05, ic_fn=ic)
        r_tf = heat3d_residual_tf(tf.constant(u, tf.float64),
                                  tf.constant(x, tf.float64),
                                  tf.constant(x, tf.float64),
                                  tf.constant(x, tf.float64),
                                  tf.constant(t, tf.float64), 0.05,
                                  ic_fn=ic).numpy()
        np.testing.assert_allclose(r_np, r_tf, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
