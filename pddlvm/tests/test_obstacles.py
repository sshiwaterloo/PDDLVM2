"""Obstacle masks, 2D spectral solver, Brinkman penalty (fast)."""
import unittest

import numpy as np

from pddlvm.numerics import ns2d_spectral as S
from pddlvm.numerics import obstacles as O


class TestObstacleFields(unittest.TestCase):
    def test_cylinder_sdf_values(self):
        sdf = O.sdf_cylinder(np.array([[0.0, 1.0]]), np.array([[0.0, 0.0]]),
                             cx=0.0, cy=0.0, D=2.0)
        self.assertAlmostEqual(sdf[0, 0], -1.0, places=12)  # center
        self.assertAlmostEqual(sdf[0, 1], 0.0, places=12)   # surface

    def test_mask_bounds_and_symmetry(self):
        n = 64
        X, Y = O.grid_coords_2d(n, n, 4.0, 4.0)
        mask = O.smooth_mask(O.sdf_cylinder(X, Y, 2.0, 2.0, 1.0), 0.1)
        self.assertTrue(np.all((mask >= 0.0) & (mask <= 1.0)))
        self.assertGreater(mask[n // 2, n // 2], 0.999)  # well inside solid
        # 90-degree rotational symmetry of the centered cylinder
        self.assertTrue(np.allclose(mask, mask[::-1, ::-1], atol=1e-12))
        self.assertTrue(np.allclose(mask, mask.T[::-1], atol=1e-12))
        # far field -> 0
        self.assertLess(mask[0, 0], 1e-3)

    def test_sphere_field_shape(self):
        pen, mask = O.sphere_field(32, 32, 32, 4.0, 4.0, 4.0,
                                   2.0, 2.0, 2.0, 1.0, 0.1, 1e-2)
        self.assertEqual(pen.shape, (32, 32, 32))
        self.assertEqual(mask.shape, (32, 32, 32))
        self.assertGreater(mask[16, 16, 16], 0.999)


class TestSolver2D(unittest.TestCase):
    def test_inviscid_tgv_energy_conservation(self):
        n = 32
        L = 2.0 * np.pi
        x = (np.arange(n) + 0.5) * L / n
        X, Y = np.meshgrid(x, x)
        u0 = np.sin(X) * np.cos(Y)
        v0 = -np.cos(X) * np.sin(Y)
        e0 = 0.5 * np.mean(u0 ** 2 + v0 ** 2)
        u, v, _, _ = S.spectral_solve_2d(u0, v0, nu=0.0, dt=0.002,
                                         n_steps=500, Lx=L, Ly=L)
        e1 = 0.5 * np.mean(u ** 2 + v ** 2)
        self.assertLess(abs(e1 - e0) / e0, 1e-8)

    def test_uniform_penalty_decay_matches_analytic(self):
        n = 32
        lam = 100.0
        pen = np.full((n, n), lam)
        u, v, _, _ = S.spectral_solve_2d(np.ones((n, n)), np.zeros((n, n)),
                                         nu=0.0, dt=0.001, n_steps=200,
                                         Lx=4.0, Ly=4.0, penalty=pen)
        ana = np.exp(-lam * 0.2)
        self.assertLess(np.abs(u - ana).max(), 1e-6)

    def test_viscous_tgv_decays_at_analytic_rate(self):
        # 2D TGV is an exact viscous NS solution: u = sin x cos y e^{-2 nu t}
        n = 64
        L = 2.0 * np.pi
        nu = 0.05
        x = (np.arange(n) + 0.5) * L / n
        X, Y = np.meshgrid(x, x)
        u0 = np.sin(X) * np.cos(Y)
        v0 = -np.cos(X) * np.sin(Y)
        t_end = 1.0
        dt = 0.01
        u, v, _, _ = S.spectral_solve_2d(u0, v0, nu=nu, dt=dt,
                                         n_steps=int(t_end / dt),
                                         Lx=L, Ly=L)
        u_ex = u0 * np.exp(-2.0 * nu * t_end)
        v_ex = v0 * np.exp(-2.0 * nu * t_end)
        self.assertLess(np.abs(u - u_ex).max(), 1e-9)
        self.assertLess(np.abs(v - v_ex).max(), 1e-9)

    def test_driven_flow_pins_mean_and_damps_solid(self):
        n = 64
        pen, mask = O.cylinder_field(n, n, 4.0, 4.0, 2.0, 2.0, 1.0,
                                     width=0.1, eps=1e-2)
        u, v, _, probes = S.spectral_solve_2d(np.ones((n, n)),
                                              np.zeros((n, n)),
                                              nu=1.0 / 100, dt=0.002,
                                              n_steps=3000, Lx=4.0, Ly=4.0,
                                              U_target=1.0, penalty=pen,
                                              probe_every=500)
        self.assertAlmostEqual(u.mean(), 1.0, places=12)   # exact pinning
        self.assertLess(np.abs(u[mask > 0.9]).max(), 1e-2)  # solid ~ at rest
        self.assertIsNotNone(probes)
        self.assertEqual(len(probes), 6)

    def test_vorticity_exact_periodic_field(self):
        n = 32
        L = 2.0 * np.pi
        x = (np.arange(n) + 0.5) * L / n
        X, Y = np.meshgrid(x, x)
        u = np.sin(Y)
        v = np.cos(X)
        w = S.vorticity(u, v, Lx=L, Ly=L)
        w_exact = -np.sin(X) - np.cos(Y)
        self.assertTrue(np.allclose(w, w_exact, atol=1e-10))


if __name__ == "__main__":
    unittest.main()
