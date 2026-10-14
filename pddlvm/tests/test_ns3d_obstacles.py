"""3D spectral obstacle solver validation (sphere, periodic box).

Mirrors the 2D obstacle validation philosophy (test_obstacles.py): exact
analytic targets for the exponential integrating factors, solid-region
suppression, and IBM probe sanity.  All runs are tiny (32^3-40^3) so the
suite stays fast on CPU.
"""

import unittest

import numpy as np

from pddlvm.numerics.ns3d_spectral import (
    q_criterion_3d,
    spectral_solve_obstacle,
)
from pddlvm.numerics.obstacles import cylinder3d_field, sphere_field


def _tgv_fields(n, L, A=1.0):
    """Single-mode Taylor-Green: div-free by construction (same layout as
    ``turbulence.tgv_initial`` -- the x-derivative of u cancels the
    y-derivative of v; naive meshgrid constructions are NOT div-free)."""
    x = np.linspace(0.0, L, n, endpoint=False)
    k = 2.0 * np.pi / L
    xg = x[None, None, :]
    yg = x[None, :, None]
    zg = x[:, None, None]
    u = A * np.sin(k * xg) * np.cos(k * yg) * np.cos(k * zg)
    v = -A * np.cos(k * xg) * np.sin(k * yg) * np.cos(k * zg)
    w = np.zeros_like(u)
    return u, v, w, k


class TestSpectralObstacle3D(unittest.TestCase):
    def test_penalty_uniform_decay(self):
        """Pure penalty decay matches exp(-lambda t) to machine precision."""
        n, L = 32, 2.0
        u0 = np.full((n, n, n), 0.7)
        v0 = np.full((n, n, n), 0.1)
        w0 = np.full((n, n, n), -0.2)
        lam0, dt, n_steps = 3.0, 0.05, 40
        penalty = np.full((n, n, n), lam0)
        u, v, w, _, _ = spectral_solve_obstacle(
            u0, v0, w0, nu=0.0, dt=dt, n_steps=n_steps, L=L, penalty=penalty)
        exact = np.exp(-lam0 * dt * n_steps)
        self.assertLess(np.abs(u.mean() - 0.7 * exact) / 0.7, 1e-12)
        self.assertLess(np.abs(v.mean() - 0.1 * exact) / 0.1, 1e-12)
        self.assertLess(np.abs(w.mean() + 0.2 * exact) / 0.2, 1e-12)

    def test_penalty_viscous_decay_combined(self):
        """A div-free shear sheet u = sin(k z) e_x (u depends on z only, so
        u.grad u = 0) decays at the analytic combined rate
        exp(-(nu k^2 + lambda) t) to machine precision -- validates the
        exact Lawson factors together."""
        n, L = 32, 2.0
        z = (np.arange(n) + 0.5) * L / n
        # depends on axis 0 (kz), x-component: div u = du/dx = 0
        u0 = np.sin(2.0 * np.pi * z / L)[:, None, None] * np.ones((1, n, n))
        v0 = np.zeros((n, n, n))
        w0 = np.zeros((n, n, n))
        nu, lam0, dt, n_steps = 0.01, 2.0, 0.05, 20
        k = 2.0 * np.pi / L
        penalty = np.full((n, n, n), lam0)
        u, v, w, _, _ = spectral_solve_obstacle(
            u0, v0, w0, nu=nu, dt=dt, n_steps=n_steps, L=L, penalty=penalty)
        exact = np.exp(-(nu * k ** 2 + lam0) * dt * n_steps)
        self.assertLess(np.max(np.abs(u - u0 * exact)), 1e-12)

    def test_inviscid_tgv_energy(self):
        """Without obstacles the Lawson scheme reduces to RK4; energy
        conservation matches the classical solver (same dt regime)."""
        n, L = 32, 2.0 * np.pi
        u0, v0, w0, _ = _tgv_fields(n, L)
        dt, n_steps = 0.004, 125

        def energy(u, v, w):
            return 0.5 * np.mean(u ** 2 + v ** 2 + w ** 2)

        e0 = energy(u0, v0, w0)
        u, v, w, _, _ = spectral_solve_obstacle(
            u0, v0, w0, nu=0.0, dt=dt, n_steps=n_steps, L=L)
        self.assertLess(abs(energy(u, v, w) - e0) / e0, 1e-4)

    def test_viscous_tgv_dissipation_rate(self):
        """Energy budget dE/dt = -2 nu Z (holds for any solution; the honest
        3D viscous check -- the single-mode 3D TGV is NOT an exact viscous
        solution, unlike its 2D counterpart)."""
        n, L = 32, 2.0 * np.pi
        nu, dt, n_steps = 0.01, 0.004, 250  # T = 1.0
        u0, v0, w0, _ = _tgv_fields(n, L)

        def energy(u, v, w):
            return 0.5 * np.mean(u ** 2 + v ** 2 + w ** 2)

        kx = np.fft.fftfreq(n, d=L / n) * 2.0 * np.pi
        kmag = np.sqrt(kx[None, None, :] ** 2 + kx[None, :, None] ** 2
                       + kx[:, None, None] ** 2)
        e0 = energy(u0, v0, w0)
        z0 = 0.5 * np.sum(kmag ** 2 *
                          (np.abs(np.fft.fftn(u0)) ** 2
                           + np.abs(np.fft.fftn(v0)) ** 2)) / n ** 6
        u, v, w, _, _ = spectral_solve_obstacle(
            u0, v0, w0, nu=nu, dt=dt, n_steps=n_steps, L=L)
        e1 = energy(u, v, w)
        z1 = 0.5 * np.sum(kmag ** 2 *
                          (np.abs(np.fft.fftn(u)) ** 2
                           + np.abs(np.fft.fftn(v)) ** 2)) / n ** 6
        measured = -(e1 - e0) / (dt * n_steps)
        expected = 2.0 * nu * 0.5 * (z0 + z1)
        self.assertLess(abs(measured - expected) / expected, 0.15)

    def test_sphere_suppression_and_probes(self):
        """Pinned flow through a sphere: solid region ~0, Cd > 0, probes
        record (step, Cd, Cl_y, Cl_z)."""
        n, L = 40, 4.0
        nu, dt, n_steps = 1.0 / 100.0, 0.02, 100
        u0 = np.full((n, n, n), 1.0)
        v0 = np.zeros((n, n, n))
        w0 = np.zeros((n, n, n))
        lam, mask = sphere_field(n, n, n, L, L, L, L / 2, L / 2, L / 2,
                                 1.0, 0.25, 0.01)
        u, v, w, _, probes = spectral_solve_obstacle(
            u0, v0, w0, nu=nu, dt=dt, n_steps=n_steps, L=L,
            U_target=1.0, penalty=lam, diameter=1.0, probe_every=10)
        solid = mask > 0.9
        self.assertLess(float(np.abs(u[solid]).max()), 0.1,
                        "solid region not suppressed")
        self.assertAlmostEqual(float(u.mean()), 1.0, places=8)
        self.assertEqual(len(probes), n_steps // 10)
        cd = np.array([p[1] for p in probes])
        self.assertGreater(cd[-1], 0.0)
        self.assertTrue(np.all(np.isfinite(cd)))


    def test_cylinder3d_geometry(self):
        """3D pillar: constant along the solver's z (array axis 0), disk
        cross-section of diameter D in the (x, y) plane.  Pins the explicit
        axis convention (axis 0=z, axis 1=y, axis 2=x)."""
        n, L = 32, 4.0
        cx = cy = L / 2
        lam, mask = cylinder3d_field(n, n, n, L, L, L, cx, cy, 1.0, 0.25, 0.01)
        # constant along axis 0 (pillar axis)
        self.assertTrue(np.allclose(mask[0], mask[n // 2]))
        self.assertTrue(np.allclose(mask[0], mask[-1]))
        # cross-section (any z-slice): solid inside r < D/2, fluid outside
        # 2D slice coords: axis 0 = y, axis 1 = x (mid = mask[0] is (ny, nx))
        x2 = ((np.arange(n) + 0.5) * L / n)[None, :]
        y2 = ((np.arange(n) + 0.5) * L / n)[:, None]
        r = np.sqrt((x2 - cx) ** 2 + (y2 - cy) ** 2)
        mid = mask[0]
        # mask ~1 well inside (r << D/2) and ~0 well outside (r >> D/2);
        # the tanh transition width is ~2*width=0.5 around r=D/2=0.5
        self.assertGreater(float(mid[r < 0.2].min()), 0.9)   # inside solid
        self.assertLess(float(mid[r > 0.8].max()), 0.1)      # outside solid
        # symmetry about the axis (two radial directions agree)
        self.assertTrue(np.allclose(mid, mid.T))


    def test_q_criterion_shear_sheet_zero(self):
        """A pure shear sheet u = sin(k z) e_x has equal vorticity and strain
        magnitude (Q = 0) -- exact analytic target for the Q-criterion helper."""
        n, L = 32, 2.0
        z = (np.arange(n) + 0.5) * L / n
        u = np.sin(2.0 * np.pi * z / L)[:, None, None] * np.ones((1, n, n))
        v = np.zeros((n, n, n))
        w = np.zeros((n, n, n))
        q = q_criterion_3d(u, v, w, L=L)
        self.assertLess(float(np.abs(q).max()), 1e-10)


if __name__ == "__main__":
    unittest.main()
