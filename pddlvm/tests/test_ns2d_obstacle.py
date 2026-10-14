"""Tests for the 2D obstacle NS residual.

Uses ``unittest`` (not pytest) so ``python -m unittest discover`` works, in
line with the rest of the repo (see docs/MATH_PLAN.md Finding F3).
"""

import unittest

import numpy as np

from pddlvm.numerics.ns2d_obstacle import (
    ns2d_obstacle_residual,
    ns2d_obstacle_residual_tf,
)


def _grid(nt=3, n=16, L=2.0 * np.pi, t_max=0.2):
    x = np.linspace(0.0, L, n, endpoint=False)
    t = np.linspace(0.0, t_max, nt)
    return x, t


class TestNs2dObstacleResidual(unittest.TestCase):

    def test_tf_matches_numpy(self):
        """The TF residual must agree with the numpy reference on a random field."""
        import tensorflow as tf

        np.random.seed(0)
        nt, n = 4, 16
        x, t = _grid(nt, n)
        u = np.random.randn(nt, n, n)
        v = np.random.randn(nt, n, n)
        p = np.random.randn(nt, n, n)
        lam = np.abs(np.random.randn(n, n))
        u0, v0 = np.random.randn(n, n), np.random.randn(n, n)

        r_np = ns2d_obstacle_residual(u, v, p, x, t, 0.01, lam, ic=(u0, v0))
        r_tf = ns2d_obstacle_residual_tf(
            tf.constant(u), tf.constant(v), tf.constant(p),
            tf.constant(x), tf.constant(t), 0.01, tf.constant(lam),
            ic=(tf.constant(u0), tf.constant(v0)))
        np.testing.assert_allclose(r_tf.numpy(), r_np, rtol=1e-5, atol=1e-6)

    def test_euler_tgv_is_exact_solution(self):
        """The steady inviscid Taylor-Green vortex leaves a residual that
        vanishes to O(dx^2) (no obstacle, nu=0)."""
        import tensorflow as tf

        nt, n = 3, 32
        x, t = _grid(nt, n)
        X, Y = np.meshgrid(x, x)
        u = np.sin(X) * np.cos(Y)
        v = -np.cos(X) * np.sin(Y)
        p = 0.25 * (np.cos(2 * X) + np.cos(2 * Y))
        lam = np.zeros((n, n))
        u = np.stack([u] * nt); v = np.stack([v] * nt); p = np.stack([p] * nt)
        r = ns2d_obstacle_residual_tf(
            tf.constant(u, tf.float32), tf.constant(v, tf.float32),
            tf.constant(p, tf.float32), tf.constant(x, tf.float32),
            tf.constant(t, tf.float32), 0.0, tf.constant(lam, tf.float32))
        self.assertLess(float(np.max(np.abs(r.numpy()))), 0.1)

    def test_penalty_term_present(self):
        """A uniform flow through a mask produces a nonzero momentum residual
        proportional to lam (Brinkman damping), and zero with lam=0."""
        nt, n = 3, 16
        x, t = _grid(nt, n)
        u = np.ones((nt, n, n))
        v = np.zeros((nt, n, n))
        p = np.zeros((nt, n, n))
        lam = np.zeros((n, n))
        lam[n // 2, n // 2] = 10.0  # a single penalized cell

        r0 = ns2d_obstacle_residual(u, v, p, x, t, 0.0, np.zeros_like(lam))
        r1 = ns2d_obstacle_residual(u, v, p, x, t, 0.0, lam)
        self.assertLess(float(np.max(np.abs(r0))), 1e-12)
        self.assertGreaterEqual(float(np.max(np.abs(r1))), 9.0)

    def test_separate_penalty(self):
        """``separate_penalty`` moves the Brinkman term into its own block."""
        nt, n = 3, 16
        x, t = _grid(nt, n)
        u = np.ones((nt, n, n))
        v = np.ones((nt, n, n))
        p = np.zeros((nt, n, n))
        lam = np.zeros((n, n))
        lam[n // 2, n // 2] = 5.0
        r_sep = ns2d_obstacle_residual(u, v, p, x, t, 0.0, lam,
                                       separate_penalty=True)
        r_mix = ns2d_obstacle_residual(u, v, p, x, t, 0.0, lam,
                                       separate_penalty=False)
        n_pen = 2 * (nt - 1) * n * n  # pen_x + pen_y
        self.assertEqual(len(r_sep), len(r_mix) + n_pen)
        # the penalty block carries lam*u = 5 at the penalized cell
        pen_block = r_sep[2 * (nt - 1) * n * n:
                          2 * (nt - 1) * n * n + n_pen]
        self.assertGreater(float(np.max(np.abs(pen_block))), 4.0)


if __name__ == "__main__":
    unittest.main()
