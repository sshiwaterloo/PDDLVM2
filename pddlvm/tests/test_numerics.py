"""MMS verification of the numerics layer.

Every solver ships with a manufactured solution:

- FEM Poisson:            u* = sin(pi x), kappa = 2 + x  (O(h^2), < 1e-8)
- Chebyshev spectral:     u* = sin(pi x), kappa const   (spectral, < 1e-10)
- Burgers transient:      u* = exp(-t) sin(2 pi x)      (space O(dx), time O(dt^2))
- Burgers steady:         u* = 1 + 0.3 x^2              (Newton, O(dx^2), < 1e-8)

Also: vectorized FEM assembly is cross-checked against an independent naive
reference, and the tf.function tracing behaviour of the assembly is asserted
(retracing warnings must not fire for a stable call signature).
"""

import logging
import unittest

import numpy as np
import tensorflow as tf
from scipy.linalg import solve_banded

from pddlvm.numerics.chebyshev import (
    cheb_T_mat,
    cheb_fit,
    poisson_const_kappa_values,
)
from pddlvm.numerics.fd import solve_burgers_steady, solve_burgers_transient
from pddlvm.numerics.fem import (
    assemble_poisson_1d,
    assemble_poisson_1d_banded,
    solve_poisson_1d,
)
from pddlvm.numerics.mms import (
    burgers_mms_source_steady,
    burgers_mms_u_steady,
    convergence_rates,
    h1_semi_error,
    l2_error,
    poisson_mms_kappa,
    poisson_mms_source,
    poisson_mms_u,
)


def _l2_error_piecewise_linear(u_nodes, u_exact, x):
    """L2 error of a piecewise-linear FEM solution.

    1D linear FEM is superconvergent at nodes (u_h(x_j) = u(x_j) to
    roundoff for exact loads), so a node-only quadrature of (u_h - u)^2 is
    degenerate.  Refining each panel with its midpoint (where u_h is the
    exact linear interpolant) turns the trapezoid rule into an
    exact-for-quadratic quadrature of the true interpolation error.
    """
    xm = 0.5 * (x[:-1] + x[1:])
    xe = np.sort(np.concatenate([x, xm]))
    uh = np.interp(xe, x, u_nodes)
    err = uh - u_exact(xe)
    return np.sqrt(np.trapz(err ** 2, xe))


class TestFemAssembly(unittest.TestCase):
    """Vectorized assembly vs an independent naive reference."""

    def test_matches_naive_assembly(self):
        x = np.linspace(-1.0, 1.0, 9)
        kappa_mid = 2.0 + (x[:-1] + x[1:]) / 2.0
        f = np.exp(x)
        bc_l, bc_r = 0.3, -0.7

        a_mat, f_vec = assemble_poisson_1d(
            x, kappa_mid, f, bc_l, bc_r, dtype=tf.float64)
        a_mat, f_vec = a_mat.numpy(), f_vec.numpy()

        n = x.shape[0]
        a_ref = np.zeros((n, n))
        f_ref = np.zeros(n)
        for e in range(n - 1):
            h = x[e + 1] - x[e]
            ke = kappa_mid[e]
            a_ref[e, e] += ke / h
            a_ref[e + 1, e + 1] += ke / h
            a_ref[e, e + 1] -= ke / h
            a_ref[e + 1, e] -= ke / h
            f_l, f_r = f[e], f[e + 1]
            f_ref[e] += h * (f_l / 3.0 + f_r / 6.0)
            f_ref[e + 1] += h * (f_l / 6.0 + f_r / 3.0)
        f_ref = f_ref - bc_l * a_ref[:, 0] - bc_r * a_ref[:, -1]
        f_ref[0], f_ref[-1] = bc_l, bc_r
        a_ref[0, :] = a_ref[-1, :] = 0.0
        a_ref[:, 0] = a_ref[:, -1] = 0.0
        a_ref[0, 0] = a_ref[-1, -1] = 1.0

        np.testing.assert_allclose(a_mat, a_ref, atol=1e-14)
        np.testing.assert_allclose(f_vec, f_ref, atol=1e-14)

    def test_banded_matches_dense(self):
        """Banded (O(n)) assembly must equal the dense assembly exactly."""
        x = np.linspace(-1.0, 1.0, 9)
        kappa_mid = 2.0 + (x[:-1] + x[1:]) / 2.0
        f = np.exp(x)
        bc_l, bc_r = 0.3, -0.7
        a_dense, f_dense = assemble_poisson_1d(
            x, kappa_mid, f, bc_l, bc_r, dtype=tf.float64)
        sub, diag, sup, f_band = assemble_poisson_1d_banded(
            x, kappa_mid, f, bc_l, bc_r, dtype=tf.float64)

        n = x.shape[0]
        a_band = np.zeros((n, n))
        a_band[1:, :-1] += np.diag(sub.numpy()[1:])
        a_band += np.diag(diag.numpy())
        a_band[:-1, 1:] += np.diag(sup.numpy()[:-1])
        np.testing.assert_allclose(a_band, a_dense.numpy(), atol=1e-14)
        np.testing.assert_allclose(f_band.numpy(), f_dense.numpy(), atol=1e-14)

    def test_dirichlet_exact(self):
        x = np.linspace(-1.0, 1.0, 33)
        kappa_mid = np.ones(32)
        u = solve_poisson_1d(x, kappa_mid, np.zeros(33), 0.3, -0.7,
                             dtype=tf.float64).numpy()
        self.assertAlmostEqual(u[0], 0.3, places=12)
        self.assertAlmostEqual(u[-1], -0.7, places=12)


class TestFemMms(unittest.TestCase):
    """h-convergence against the manufactured solution (float64)."""

    def test_convergence_rate_and_error(self):
        errs_l2, errs_h1, hs = [], [], []
        for ne in (1024, 2048, 4096, 8192, 16384, 32768):
            # perturbed mesh: breaks the nodal superconvergence of uniform
            # linear FEM (which would make the node-quadrature L2 error
            # identically zero) so the O(h^2) rate is a real measurement
            h_unif = 2.0 / ne
            j = np.arange(ne + 1)
            x = -1.0 + 2.0 * j / ne + 0.1 * h_unif * np.sin(2 * np.pi * j / ne)
            kappa_mid = poisson_mms_kappa((x[:-1] + x[1:]) / 2.0)
            sub, diag, sup, f_vec = assemble_poisson_1d_banded(
                x, kappa_mid, poisson_mms_source(x),
                poisson_mms_u(-1.0), poisson_mms_u(1.0), dtype=tf.float64)
            ab = np.vstack([np.concatenate([[0.0], sup.numpy()[:-1]]),
                            diag.numpy(),
                            np.concatenate([sub.numpy()[1:], [0.0]])])
            u = solve_banded((1, 1), ab, f_vec.numpy())
            errs_l2.append(_l2_error_piecewise_linear(
                u, poisson_mms_u, x))
            errs_h1.append(h1_semi_error(u, poisson_mms_u(x), x))
            hs.append(h_unif)

        rates_l2 = convergence_rates(errs_l2, hs)
        rates_h1 = convergence_rates(errs_h1, hs)
        print(f"FEM MMS: h={hs}, L2={np.round(errs_l2, 2)}, "
              f"L2 rates={np.round(rates_l2, 2)}, H1 rates={np.round(rates_h1, 2)}")
        self.assertGreater(rates_l2[-1], 1.8, "expected O(h^2) L2 convergence")
        self.assertGreater(rates_h1[-1], 0.9, "expected O(h) H1 convergence")
        self.assertLess(errs_l2[-1], 1e-8)

    def test_dense_solve_path(self):
        # the tf.linalg.solve path used inside the training graph, at a
        # coarse resolution (dense solve would be too slow at fine meshes)
        x = np.linspace(-1.0, 1.0, 65)
        kappa_mid = poisson_mms_kappa((x[:-1] + x[1:]) / 2.0)
        u = solve_poisson_1d(x, kappa_mid, poisson_mms_source(x),
                             poisson_mms_u(-1.0), poisson_mms_u(1.0),
                             dtype=tf.float64).numpy()
        self.assertLess(l2_error(u, poisson_mms_u(x), x), 5e-3)


class TestChebyshev(unittest.TestCase):
    """Chebyshev utilities and the spectral Poisson solver."""

    def test_T_mat_matches_formula(self):
        x = tf.constant(np.linspace(-0.99, 0.99, 17), tf.float64)
        T = cheb_T_mat(8, x, dtype=tf.float64).numpy()
        ref = np.cos(np.arange(8)[:, None] * np.arccos(x.numpy()))
        np.testing.assert_allclose(T, ref, atol=1e-12)

    def test_spectral_poisson_mms(self):
        # -2 u'' = 2 pi^2 sin(pi x), u(-1) = u(1) = 0 -> u* = sin(pi x)
        n_f = 32
        nodes = np.cos(np.pi * np.arange(2 * n_f + 1) / (2 * n_f))
        f_vals = tf.constant(2.0 * np.pi ** 2 * np.sin(np.pi * nodes),
                             tf.float64)
        f_coeffs = cheb_fit(tf.constant(nodes, tf.float64), f_vals, n_f,
                            dtype=tf.float64)
        u = poisson_const_kappa_values(
            f_coeffs, tf.constant(nodes, tf.float64),
            kappa=2.0, bc_left=0.0, bc_right=0.0, dtype=tf.float64).numpy()
        err = np.max(np.abs(u - np.sin(np.pi * nodes)))
        print(f"Chebyshev spectral MMS: max err = {err:.2e} (n_f={n_f})")
        self.assertLess(err, 1e-10)

    def test_spectral_convergence(self):
        errors = []
        for n_f in (8, 12, 16, 20):
            nodes = np.cos(np.pi * np.arange(2 * n_f + 1) / (2 * n_f))
            f_vals = tf.constant(2.0 * np.pi ** 2 * np.sin(np.pi * nodes),
                                 tf.float64)
            f_coeffs = cheb_fit(tf.constant(nodes, tf.float64), f_vals, n_f,
                                dtype=tf.float64)
            u = poisson_const_kappa_values(
                f_coeffs, tf.constant(nodes, tf.float64),
                kappa=2.0, bc_left=0.0, bc_right=0.0,
                dtype=tf.float64).numpy()
            errors.append(np.max(np.abs(u - np.sin(np.pi * nodes))))
        self.assertLess(errors[-1], errors[0] * 1e-6,
                        "expected spectral (exponential) decay")
        self.assertLess(errors[-1], 1e-9)


def _weak_mms(nu, amp):
    """Diffusion-dominated transient MMS: u* = amp * exp(-t) sin(2 pi x).

    The amplitude is tiny (1e-4) so the first-order upwind space error is
    orders of magnitude below the theta-scheme time error; this isolates
    the time convergence order of the integrator.
    """
    def u(x, t):
        return amp * np.exp(-t) * np.sin(2.0 * np.pi * x)

    def s(x, t):
        e = np.exp(-t)
        return (-amp * e * np.sin(2.0 * np.pi * x)
                + amp * e * np.sin(2.0 * np.pi * x)
                * amp * e * 2.0 * np.pi * np.cos(2.0 * np.pi * x)
                + nu * amp * e * 4.0 * np.pi ** 2 * np.sin(2.0 * np.pi * x))

    return u, s


class TestBurgersTransient(unittest.TestCase):
    """MMS rates for the theta-IMEX integrator."""

    def test_space_order(self):
        from pddlvm.numerics.mms import (burgers_mms_source_transient,
                                         burgers_mms_u_transient)
        nu, t1, dt = 0.1, 0.5, 4e-5
        errs, hs = [], []
        for n in (32, 64, 128, 256):
            x = np.linspace(0.0, 1.0, n + 1)
            u0 = burgers_mms_u_transient(x, 0.0, nu)
            u, _, _ = solve_burgers_transient(
                x, u0, nu,
                source=lambda xx, t: burgers_mms_source_transient(xx, t, nu),
                t0=0.0, t1=t1, dt=dt,
                bc_left=lambda t: burgers_mms_u_transient(0.0, t, nu),
                bc_right=lambda t: burgers_mms_u_transient(1.0, t, nu))
            errs.append(l2_error(u, burgers_mms_u_transient(x, t1, nu), x))
            hs.append(x[1] - x[0])
        rates = convergence_rates(errs, hs)
        print(f"Burgers transient space rates: {np.round(rates, 3)} "
              f"(errors {np.round(errs, 5)})")
        self.assertGreater(rates[-1], 0.8)
        self.assertLess(rates[-1], 1.25)

    def test_time_order(self):
        # diffusion-dominated regime with a tiny advection amplitude so the
        # space error (O(amp^2 dx), dx = 50 dt^2) stays far below the time
        # error (O(amp dt^2) for theta=0.5, O(amp dt) for theta=1).
        nu, t1, amp = 1.0, 0.5, 1e-4
        u_mms, s_mms = _weak_mms(nu, amp)

        errs_by_theta = {}
        rates_by_theta = {}
        for theta in (0.5, 1.0):
            errs, dts = [], []
            for k in range(4):
                dt = 2.0 ** k / 400.0
                dx = 50.0 * dt ** 2
                n = max(int(round(1.0 / dx)), 8)
                x = np.linspace(0.0, 1.0, n + 1)
                u0 = u_mms(x, 0.0)
                u, _, _ = solve_burgers_transient(
                    x, u0, nu,
                    source=lambda xx, t: s_mms(xx, t),
                    t0=0.0, t1=t1, dt=dt, theta=theta,
                    bc_left=lambda t: u_mms(0.0, t),
                    bc_right=lambda t: u_mms(1.0, t))
                errs.append(l2_error(u, u_mms(x, t1), x))
                dts.append(dt)
            rates = convergence_rates(errs, dts)
            errs_by_theta[theta] = errs
            rates_by_theta[theta] = rates
            print(f"Burgers transient time errors (theta={theta}): "
                  f"{np.round(errs, 9)}, rates {np.round(rates, 3)}")

        # theta=0.5 must be at least second order in time (the measured rate
        # can exceed 2 for this mode because of cancellation); theta=1 is
        # backward Euler, so its cleanest (small-dt) rate is first-order.
        # At larger dt an O(theta*|u|*dt*dx) implicit-upwind viscosity term
        # pollutes both, so the magnitude cross-check is not asserted.
        self.assertGreater(rates_by_theta[0.5][-1], 1.9)
        self.assertLess(rates_by_theta[1.0][0], 1.5)
        # backward-Euler errors must shrink monotonically with dt
        self.assertTrue(np.all(np.diff(errs_by_theta[1.0]) > 0))


class TestBurgersSteady(unittest.TestCase):
    """Newton steady solver MMS (second-order central differences)."""

    def test_mms(self):
        nu = 0.1
        errs, hs = [], []
        for n in (512, 1024, 2048, 4096, 8192):
            x = np.linspace(0.0, 1.0, n + 1)
            u = solve_burgers_steady(
                x, nu, lambda xx: burgers_mms_source_steady(xx, nu),
                bc_left=burgers_mms_u_steady(0.0, nu),
                bc_right=burgers_mms_u_steady(1.0, nu))
            errs.append(l2_error(u, burgers_mms_u_steady(x, nu), x))
            hs.append(x[1] - x[0])
        rates = convergence_rates(errs, hs)
        print(f"Burgers steady rates: {np.round(rates, 3)} "
              f"(errors {np.round(errs, 6)})")
        self.assertGreater(rates[-1], 1.7)
        self.assertLess(errs[-1], 1e-8)
        # BCs exact to Newton tolerance
        x = np.linspace(0.0, 1.0, 1025)
        u = solve_burgers_steady(
            x, nu, lambda xx: burgers_mms_source_steady(xx, nu),
            bc_left=burgers_mms_u_steady(0.0, nu),
            bc_right=burgers_mms_u_steady(1.0, nu))
        self.assertAlmostEqual(u[0], 1.0, places=10)
        self.assertAlmostEqual(u[-1], 1.4, places=10)


class TestTfFunctionNoRetracing(unittest.TestCase):
    """The assembly must not retrace for a stable call signature."""

    def test_no_retracing(self):
        logger = logging.getLogger("tensorflow")

        class Capture(logging.Handler):
            def __init__(self):
                super().__init__(level=logging.WARNING)
                self.messages = []

            def emit(self, record):
                self.messages.append(record.getMessage())

        cap = Capture()
        logger.addHandler(cap)
        try:
            fn = tf.function(assemble_poisson_1d)
            x = tf.constant(np.linspace(-1.0, 1.0, 9), tf.float32)
            k = tf.ones(8, tf.float32)
            f = tf.ones(9, tf.float32)
            for _ in range(6):
                fn(x, k, f, 0.0, 0.0)
            retraces = [m for m in cap.messages if "retracing" in m.lower()]
            self.assertEqual(len(retraces), 0, f"retracing fired: {retraces}")
        finally:
            logger.removeHandler(cap)


class TestBurgersSpaceTime(unittest.TestCase):
    """Space-time residual operator (for the time-dependent PDDLVM demo)."""

    def test_residual_vanishes_up_to_truncation(self):
        from pddlvm.numerics.fd import burgers_space_time_residual
        from pddlvm.numerics.mms import (burgers_mms_source_transient,
                                         burgers_mms_u_transient)
        nu = 0.1

        def residual_for(nx, nt):
            x = np.linspace(0.0, 1.0, nx)
            t = np.linspace(0.0, 0.5, nt)
            # u_grid is [Nt, Nx]: first axis is time
            tt, xx = np.meshgrid(t, x, indexing="ij")
            u = burgers_mms_u_transient(xx, tt, nu)
            r = burgers_space_time_residual(
                u, x, t, nu,
                ic=lambda xv: burgers_mms_u_transient(xv, 0.0, nu),
                bc_left=lambda tv: burgers_mms_u_transient(0.0, tv, nu),
                bc_right=lambda tv: burgers_mms_u_transient(1.0, tv, nu),
                source=lambda xv, tv: burgers_mms_source_transient(
                    xv, tv, nu))
            return np.max(np.abs(r))

        r_coarse = residual_for(65, 65)
        r_fine = residual_for(129, 129)
        print(f"space-time residual MMS: coarse {r_coarse:.2e}, "
              f"fine {r_fine:.2e}")
        # truncation is O(dt + dx^2): halving both reduces it 2x-4x
        self.assertGreater(r_coarse / r_fine, 1.5)
        self.assertLess(r_coarse / r_fine, 4.0)
        self.assertLess(r_fine, 2e-2)

    def test_exact_solution_gives_zero(self):
        from pddlvm.numerics.fd import burgers_space_time_residual
        # u = 0 solves the homogeneous equation with zero IC/BCs exactly
        u = np.zeros((8, 8))
        x = np.linspace(0.0, 1.0, 8)
        t = np.linspace(0.0, 1.0, 8)
        r = burgers_space_time_residual(
            u, x, t, nu=0.1, ic=lambda xv: np.zeros_like(xv),
            bc_left=0.0, bc_right=0.0)
        self.assertEqual(np.max(np.abs(r)), 0.0)

    def test_tf_matches_numpy(self):
        from pddlvm.numerics.fd import (burgers_space_time_residual,
                                        burgers_space_time_residual_tf)
        rng = np.random.default_rng(3)
        x = np.linspace(0.0, 1.0, 17)
        t = np.linspace(0.0, 0.5, 9)
        u = rng.normal(size=(9, 17))

        def ic(xv):
            return np.sin(np.pi * xv)

        r_np = burgers_space_time_residual(u, x, t, 0.1, ic=ic,
                                           bc_left=0.2, bc_right=-0.1)
        r_tf = burgers_space_time_residual_tf(
            tf.constant(u, tf.float64), tf.constant(x, tf.float64),
            tf.constant(t, tf.float64), 0.1, ic_fn=ic,
            bc_left=0.2, bc_right=-0.1).numpy()
        np.testing.assert_allclose(r_np, r_tf, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
