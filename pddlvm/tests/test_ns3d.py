"""3D Navier-Stokes operators, spectral solver, and
the Conv3D decoder.

Gates:
- Ethier-Steinman MMS: the residual of the analytic 3D NS solution is
  O(dt + dx^2) under refinement.
- numpy/TensorFlow operator parity, exact zero solution.
- Taylor-Green initial energy E0 = 1/8 and enstrophy Z0 = 3/8 (analytic).
- Spectral solver: inviscid TGV conserves energy; viscous runs obey
  dE/dt = -2 nu Z.
- ConvFieldNet: shapes, seed reproducibility, gradient flow.
"""

import unittest

import numpy as np
import tensorflow as tf

from pddlvm.NeuralNet import ConvFieldNet
from pddlvm.numerics.mms import es3d_fields, es3d_grid
from pddlvm.numerics.ns3d import ns3d_residual, ns3d_residual_tf
from pddlvm.numerics.ns3d_spectral import spectral_solve
from pddlvm.numerics.turbulence import (
    enstrophy,
    kinetic_energy,
    tgv_initial,
)


def _es_bc_ic(n, nt, nu=0.1):
    x = np.linspace(-1.0, 1.0, n)
    t = np.linspace(0.0, 0.2, nt)
    u, v, w, p = es3d_grid(x, t, nu)
    x4 = x[None, None, None, :]
    y4 = x[None, None, :, None]
    z4 = x[None, :, None, None]

    def ic(xv, yv, zv):
        return es3d_fields(xv, yv, zv, 0.0, nu=nu)[:3]

    return x, t, u, v, w, p, ic


class TestNs3dResidual(unittest.TestCase):
    def test_es_mms_truncation_scaling(self):
        x_c, t_c, uc, vc, wc, pc, ic = _es_bc_ic(9, 9)
        x_f, t_f, uc_f, vf, wf, pf, _ = _es_bc_ic(17, 17)
        ic_f = ic  # same callable

        def residual(x, t, u, v, w, p, ic_fn):
            r = ns3d_residual(u, v, w, p, x, t, nu=0.1,
                              mode="dirichlet", ic=ic_fn,
                              bc=(u, v, w))
            return np.max(np.abs(r))

        r_coarse = residual(x_c, t_c, uc, vc, wc, pc, ic)
        r_fine = residual(x_f, t_f, uc_f, vf, wf, pf, ic_f)
        print(f"ES MMS residual: coarse {r_coarse:.2e}, fine {r_fine:.2e}")
        self.assertGreater(r_coarse / r_fine, 2.5)
        self.assertLess(r_coarse / r_fine, 4.5)
        self.assertLess(r_fine, 5e-2)

    def test_zero_solution(self):
        n, nt = 6, 4
        x = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
        t = np.linspace(0.0, 0.3, nt)
        zero = np.zeros((nt, n, n, n))
        r = ns3d_residual(zero, zero, zero, zero, x, t, nu=0.1,
                          mode="periodic")
        self.assertEqual(np.max(np.abs(r)), 0.0)
        r_tf = ns3d_residual_tf(
            tf.constant(zero, tf.float64), tf.constant(zero, tf.float64),
            tf.constant(zero, tf.float64), tf.constant(zero, tf.float64),
            tf.constant(x, tf.float64), tf.constant(t, tf.float64), 0.1,
            mode="periodic").numpy()
        self.assertEqual(np.max(np.abs(r_tf)), 0.0)

    def test_tf_matches_numpy(self):
        rng = np.random.default_rng(7)
        n, nt = 6, 4
        x = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
        t = np.linspace(0.0, 0.3, nt)
        shape = (nt, n, n, n)
        fields = [rng.normal(size=shape) for _ in range(4)]
        r_np = ns3d_residual(*fields, x, t, nu=0.1, mode="periodic")
        r_tf = ns3d_residual_tf(
            *[tf.constant(f, tf.float64) for f in fields],
            tf.constant(x, tf.float64), tf.constant(t, tf.float64), 0.1,
            mode="periodic").numpy()
        np.testing.assert_allclose(r_np, r_tf, atol=1e-12)


class TestTgvDiagnostics(unittest.TestCase):
    def test_initial_energy_and_enstrophy(self):
        u0, v0, w0 = tgv_initial(24)
        # E0 = 0.5 * (<sin^2><cos^2><cos^2> + <cos^2><sin^2><cos^2>) = 1/8
        self.assertAlmostEqual(kinetic_energy(u0, v0, w0), 0.125, places=4)
        # Z0 = 0.5 * (1/8 + 1/8 + 4/8) = 3/8
        self.assertAlmostEqual(enstrophy(u0, v0, w0), 0.375, places=4)


class TestSpectralSolver(unittest.TestCase):
    def test_inviscid_energy_conservation(self):
        u0, v0, w0 = tgv_initial(32)
        e0 = kinetic_energy(u0, v0, w0)
        u, v, w, _ = spectral_solve(u0, v0, w0, nu=0.0, dt=0.004,
                                    n_steps=125)
        e1 = kinetic_energy(u, v, w)
        drift = abs(e1 - e0) / e0
        print(f"inviscid TGV energy drift: {drift:.2e}")
        self.assertLess(drift, 1e-4)

    def test_viscous_dissipation_rate(self):
        nu = 0.01
        u0, v0, w0 = tgv_initial(32)
        e0 = kinetic_energy(u0, v0, w0)
        z0 = enstrophy(u0, v0, w0)
        u, v, w, _ = spectral_solve(u0, v0, w0, nu=nu, dt=0.004,
                                    n_steps=250)  # T = 1.0
        e1 = kinetic_energy(u, v, w)
        z1 = enstrophy(u, v, w)
        measured = -(e1 - e0) / 1.0
        expected = 2.0 * nu * 0.5 * (z0 + z1)
        rel = abs(measured - expected) / expected
        print(f"dissipation check: measured {measured:.3e}, "
              f"2 nu <Z> {expected:.3e}, rel {rel:.2e}")
        self.assertLess(rel, 0.15)


class TestConvFieldNet(unittest.TestCase):
    def test_shapes(self):
        net = ConvFieldNet("c", z_dim=2, field_shape=(16, 16, 16),
                           channels=1, base_filters=8, init_seed=1)
        z = tf.random.normal([4, 2])
        mean, logvar = net.Map(z)
        self.assertEqual(tuple(mean.shape), (4, 16, 16, 16, 1))
        self.assertEqual(tuple(logvar.shape), (4, 16, 16, 16, 1))
        self.assertTrue(bool(tf.reduce_all(tf.math.is_finite(mean))))

    def test_multi_channel(self):
        net = ConvFieldNet("c3", z_dim=4, field_shape=(8, 8, 8),
                           channels=3, base_filters=8, init_seed=2)
        mean, logvar = net.Map([tf.random.normal([2, 4])])
        self.assertEqual(tuple(mean.shape), (2, 8, 8, 8, 3))

    def test_seed_reproducibility(self):
        n1 = ConvFieldNet("a", 2, (8, 8, 8), init_seed=7)
        n2 = ConvFieldNet("b", 2, (8, 8, 8), init_seed=7)
        for w1, w2 in zip(n1.trainable_variables, n2.trainable_variables):
            np.testing.assert_array_equal(w1.numpy(), w2.numpy())

    def test_gradient_flows(self):
        net = ConvFieldNet("g", 2, (8, 8, 8), base_filters=4, init_seed=3)
        z = tf.random.normal([2, 2])
        with tf.GradientTape() as tape:
            mean, _ = net.Map(z)
            loss = tf.reduce_sum(mean ** 2)
        grads = tape.gradient(loss, net.trainable_variables)
        self.assertTrue(all(g is not None for g in grads))
        self.assertTrue(any(float(tf.reduce_max(tf.abs(g))) > 0
                            for g in grads))

    def test_non_power_of_two_raises(self):
        with self.assertRaises(ValueError):
            ConvFieldNet("bad", 2, (12, 12, 12))


if __name__ == "__main__":
    unittest.main()
