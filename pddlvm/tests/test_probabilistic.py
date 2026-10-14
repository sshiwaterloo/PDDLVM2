"""Tests for the probabilistic layer.

Covers:
- distribution helpers against hand-computed values,
- reparameterization-gradient unbiasedness (analytic E[f] vs Monte Carlo
  mean of the pathwise gradient),
- the physics Gaussian likelihood against the legacy formula,
- per-component ELBO accounting,
- the legacy VE facade (prior/post paths, configuration errors),
- the Trainer (loss decreases, LR schedule, early stopping) and
  checkpoint/resume with a seamless loss curve.
"""

import os
import tempfile
import unittest

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from pddlvm.numerics.fem import assemble_poisson_1d
from pddlvm.probabilistic import (
    Checkpointer,
    Trainer,
    build_dataset,
    combine_terms,
    diag_gaussian_log_prob,
    gaussian_residual_log_likelihood,
    physics_gaussian_log_likelihood,
    reparameterize,
)

tfd = tfp.distributions


def _finite(t):
    return bool(tf.reduce_all(tf.math.is_finite(t)))


class TestDistributions(unittest.TestCase):
    """Distribution helpers vs hand-computed values."""

    def test_diag_gaussian_log_prob_matches_manual(self):
        x = tf.constant([[0.5, -1.0]], tf.float64)
        mean = tf.constant([[0.0, 0.2]], tf.float64)
        logvar = tf.constant([[0.0, 0.5]], tf.float64)
        got = diag_gaussian_log_prob(x, mean, logvar, axis=-1).numpy()
        want = sum(
            -0.5 * ((xi - m) ** 2 * np.exp(-lv) + lv + np.log(2 * np.pi))
            for xi, m, lv in zip(x.numpy()[0], mean.numpy()[0],
                                 logvar.numpy()[0]))
        self.assertAlmostEqual(got[0], want, places=12)

    def test_reparameterize_shape_and_seed(self):
        mean = tf.zeros((4, 3))
        logvar = tf.zeros((4, 3))
        # note: a per-call seed alone is not reproducible on this TF build
        # (it mixes with a global counter); reproducibility is guaranteed by
        # resetting the global seed, which is what training runs rely on.
        tf.random.set_seed(7)
        z1 = reparameterize(mean, logvar)
        tf.random.set_seed(7)
        z2 = reparameterize(mean, logvar)
        self.assertEqual(tuple(z1.shape), (4, 3))
        np.testing.assert_array_equal(z1.numpy(), z2.numpy())

    def test_residual_likelihood_matches_manual(self):
        r = tf.constant([[1.0, 2.0, -3.0]], tf.float64)
        sigma = tf.constant(0.7, tf.float64)
        got = gaussian_residual_log_likelihood(r, sigma).numpy()[0]
        want = -0.5 * (14.0 / 0.49 + 3.0 * np.log(2 * np.pi * 0.49))
        self.assertAlmostEqual(got, want, places=12)


class TestReparameterizationGradient(unittest.TestCase):
    """Pathwise (reparameterization) gradient is unbiased."""

    def test_gradient_unbiased(self):
        # f(z) = a z^2 + b z, z = mu + sigma * eps:
        #   E[f] = a (mu^2 + sigma^2) + b mu
        #   d/dm E[f] = 2 a mu + b            d/dlv E[f] = a exp(lv)
        # Monte Carlo pathwise gradients must converge to those values.
        a, b = 1.5, -0.7
        mu0, lv0 = 0.3, np.log(0.5)  # sigma = 0.5
        n_chunks, chunk = 20, 10000

        mu = tf.Variable(mu0, dtype=tf.float32)
        lv = tf.Variable(lv0, dtype=tf.float32)
        grad_mu_list, grad_lv_list = [], []
        for k in range(n_chunks):
            with tf.GradientTape() as tape:
                eps = tf.random.normal([chunk], seed=1000 + k)
                z = mu + tf.exp(lv * 0.5) * eps
                f = tf.reduce_mean(a * z ** 2 + b * z)
            gm, glv = tape.gradient(f, [mu, lv])
            grad_mu_list.append(gm)
            grad_lv_list.append(glv)
        mc_mu = float(tf.reduce_mean(grad_mu_list))
        mc_lv = float(tf.reduce_mean(grad_lv_list))

        want_mu = 2 * a * mu0 + b
        want_lv = a * np.exp(lv0)
        # std of the pathwise gradient ~ 2 a sigma / sqrt(N) for mu,
        # a sigma^2 sqrt(2) / sqrt(N) for lv; allow 6 sigma.
        std_mu = 2 * a * 0.5 / np.sqrt(n_chunks * chunk)
        std_lv = a * 0.25 * np.sqrt(2) / np.sqrt(n_chunks * chunk)
        print(f"reparam gradient MC: mu {mc_mu:.4f} (want {want_mu:.4f}), "
              f"lv {mc_lv:.4f} (want {want_lv:.4f})")
        self.assertAlmostEqual(mc_mu, want_mu, delta=6 * std_mu)
        self.assertAlmostEqual(mc_lv, want_lv, delta=6 * std_lv)


class TestPhysicsLikelihood(unittest.TestCase):
    """Gaussian physics likelihood vs the hand-computed formula."""

    def test_matches_manual(self):
        x = np.linspace(-1.0, 1.0, 17)
        kappa = np.ones(16)
        f = np.exp(x)
        a_mat, f_vec = assemble_poisson_1d(x, kappa, f, 0.1, -0.2,
                                           dtype=tf.float64)
        u = tf.constant(np.sin(x) * 0.3, tf.float64)
        sigma = tf.constant(0.4, tf.float64)
        got = physics_gaussian_log_likelihood(a_mat, f_vec, u, sigma).numpy()
        d = a_mat @ u[:, None] - f_vec[:, None]
        n = a_mat.shape[0]
        want = -0.5 * (n * np.log(0.4 ** 2)
                       + float(tf.norm(d) ** 2) / 0.4 ** 2
                       + np.log(2 * np.pi))
        self.assertAlmostEqual(float(got), want, places=10)


class TestElboTerms(unittest.TestCase):
    """Per-component accounting arithmetic."""

    def test_combine_terms(self):
        L = tf.constant([1.0, 2.0, 3.0])
        P = tf.constant([0.5, 0.5, 0.5])
        Q = tf.constant([0.2, 0.1, 0.3])
        terms = combine_terms(L, P, Q)
        want_elbo = float(tf.reduce_mean(L + P - Q))
        self.assertAlmostEqual(float(terms.elbo), want_elbo, places=6)
        self.assertAlmostEqual(float(terms.nelbo), -want_elbo, places=6)
        self.assertAlmostEqual(float(terms.log_likelihood),
                               float(tf.reduce_mean(L)), places=6)
        self.assertAlmostEqual(float(terms.log_prior),
                               float(tf.reduce_mean(P)), places=6)
        self.assertAlmostEqual(float(terms.log_q),
                               float(tf.reduce_mean(Q)), places=6)
        d = terms.as_dict()
        self.assertAlmostEqual(d["elbo"], want_elbo, places=6)


class TestVEModelCompat(unittest.TestCase):
    """The legacy VE facade still works and now fails loudly when unconfigured."""

    def _make_ve(self, dimY=None):
        from pddlvm.VEModel import VE
        from pddlvm.poisson1DPDEState import PDE_State
        from pddlvm.poisson1DSolver import Solver
        x_fe = tf.constant(np.linspace(-1.0, 1.0, 21), tf.float32)
        pde = PDE_State(xFE=x_fe, dimK=1, dimU=3, dimF=1)
        solver = Solver(pde, x_fe, isNonLinear=False)
        ve = VE(pde, solver)
        if dimY is not None:
            ve.dimY = dimY
        ve.initPriorNetworks()
        ve.initPostNetworks()
        return ve

    def test_prior_mode_requires_priors(self):
        ve = self._make_ve()
        with self.assertRaises(ValueError):
            ve.compute_nelbo_prior(0)

    def test_post_mode_requires_data(self):
        ve = self._make_ve()
        with self.assertRaises(ValueError):
            ve.compute_nelbo_post(0)

    def test_prior_mode_elbo_finite_and_components(self):
        ve = self._make_ve()
        ve.setPriors(
            kappa=tfd.Normal(0.1, 0.01),
            forcing=tfd.Normal(5.0, 0.1),
            bl=tfd.Normal(0.0, 0.01),
            br=tfd.Normal(0.5, 0.01))
        terms = ve.elbo_terms_prior(0)
        self.assertTrue(_finite(terms.elbo))
        self.assertTrue(_finite(terms.log_likelihood))
        self.assertTrue(_finite(terms.log_prior))
        self.assertTrue(_finite(terms.log_q))
        want = float(terms.log_likelihood + terms.log_prior - terms.log_q)
        self.assertAlmostEqual(float(terms.elbo), want, places=4)
        # legacy scalar interface returns a finite NELBO (a fresh stochastic
        # evaluation, so only finiteness is asserted)
        nelbo = ve.compute_nelbo_prior(0)
        self.assertTrue(_finite(nelbo))

    def test_post_mode_elbo_finite(self):
        # observations live on the FE nodes: dimY must equal the node count
        # (the posterior-mode data likelihood compares y_i against uFE)
        ve = self._make_ve(dimY=21)
        n = 2
        ve.dataY = tf.constant(np.random.default_rng(0).normal(
            0, 0.5, (n, 21)), tf.float32)
        ve.dataForcingPrior = (
            tf.constant(np.zeros((n, ve.dimF), np.float32)),
            tf.constant(np.zeros((n, ve.dimF), np.float32)))
        ve.dataKappaPrior = (
            tf.constant(np.zeros((n, ve.dimK), np.float32)),
            tf.constant(np.zeros((n, ve.dimK), np.float32)))
        ve.dataBlPrior = (tf.constant(np.zeros((n, 1), np.float32)),
                          tf.constant(np.zeros((n, 1), np.float32)))
        ve.dataBrPrior = (tf.constant(np.zeros((n, 1), np.float32)),
                          tf.constant(np.zeros((n, 1), np.float32)))
        terms = ve.elbo_terms_post(0)
        self.assertTrue(_finite(terms.elbo))
        want = float(terms.log_likelihood + terms.log_prior - terms.log_q)
        self.assertAlmostEqual(float(terms.elbo), want, places=4)
        # legacy scalar interface returns the same step's NELBO
        nelbo = ve.compute_nelbo_post(0)
        self.assertTrue(_finite(nelbo))

    def test_physics_penalty_matches_shared_formula(self):
        from pddlvm.VEModel import VE
        from pddlvm.poisson1DPDEState import PDE_State
        from pddlvm.poisson1DSolver import Solver
        x_fe = tf.constant(np.linspace(-1.0, 1.0, 21), tf.float32)
        pde = PDE_State(xFE=x_fe, dimK=1, dimU=3, dimF=1)
        solver = Solver(pde, x_fe, isNonLinear=False)
        ve = VE(pde, solver)
        u = tf.reshape(tf.linspace(0.0, 1.0, 21), [1, -1])
        eps = tf.constant(0.05, tf.float32)
        got = ve.physicsPenalty(u, eps)
        a_mat, f_vec = solver.assemble_tf(tf.reshape(u, [-1]))
        want = physics_gaussian_log_likelihood(a_mat, f_vec, u[0, :], eps)
        self.assertAlmostEqual(float(got), float(want), places=4)


class TestTrainer(unittest.TestCase):
    """Trainer: convergence, LR schedule, early stop, checkpoint resume."""

    @staticmethod
    def _regression_setup():
        """Linear regression forward: nelbo = MSE(y - w x - b)."""
        rng = np.random.default_rng(0)
        x_data = rng.normal(size=(64, 1)).astype(np.float32)
        y_data = (2.5 * x_data + 0.7).astype(np.float32)
        w = tf.Variable(0.0, dtype=tf.float32)
        b = tf.Variable(0.0, dtype=tf.float32)

        def forward(batch):
            xx, yy = batch[:, 0:1], batch[:, 1:2]
            pred = w * xx + b
            r = yy - pred
            log_lik = -tf.reduce_mean(r ** 2)
            terms = combine_terms(log_lik,
                                  tf.zeros_like(log_lik),
                                  tf.zeros_like(log_lik))
            return terms

        data = np.concatenate([x_data, y_data], axis=1)
        return forward, [w, b], data

    def test_loss_decreases_and_schedule(self):
        forward, variables, data = self._regression_setup()
        opt = tf.keras.optimizers.Adam(0.05)

        def schedule(step):
            return 0.05 if step < 150 else 0.005

        trainer = Trainer(forward, variables, opt, lr_schedule=schedule)
        ds = build_dataset(data, batch_size=16).repeat()
        trainer.fit(ds, steps=300, log_every=300)
        final = trainer.history["nelbo"][-1]
        first = trainer.history["nelbo"][0]
        print(f"regression nelbo: {first:.4f} -> {final:.6f}")
        self.assertLess(final, 1e-4)
        self.assertLess(final, first)
        lrs = trainer.history["lr"]
        self.assertAlmostEqual(lrs[0], 0.05, places=5)
        self.assertAlmostEqual(lrs[-1], 0.005, places=5)
        for name in ("log_likelihood", "log_prior", "log_q", "elbo", "nelbo"):
            self.assertEqual(len(trainer.history[name]), 300)

    def test_checkpoint_resume_seamless(self):
        forward, variables, data = self._regression_setup()
        with tempfile.TemporaryDirectory() as tmp:
            opt1 = tf.keras.optimizers.Adam(0.05)
            ck = Checkpointer(os.path.join(tmp, "ckpt"),
                              optimizer=opt1, w=variables[0], b=variables[1])
            t1 = Trainer(forward, variables, opt1, checkpointer=ck,
                         checkpoint_every=100)
            t1.fit(build_dataset(data, batch_size=16).repeat(),
                   steps=100)
            # evaluate on a FIXED batch so the resume comparison is
            # apples-to-apples (training batches are shuffled)
            val_batch = tf.constant(data[:16], tf.float32)
            saved_loss = float(t1.evaluate(val_batch).nelbo)
            ck.save()

            # fresh optimizer + variables, restore, continue
            w2 = tf.Variable(0.0, dtype=tf.float32)
            b2 = tf.Variable(0.0, dtype=tf.float32)

            def forward2(batch):
                xx, yy = batch[:, 0:1], batch[:, 1:2]
                pred = w2 * xx + b2
                r = yy - pred
                log_lik = -tf.reduce_mean(r ** 2)
                return combine_terms(log_lik, tf.zeros_like(log_lik),
                                     tf.zeros_like(log_lik))

            opt2 = tf.keras.optimizers.Adam(0.05)
            ck2 = Checkpointer(os.path.join(tmp, "ckpt"),
                               optimizer=opt2, w=w2, b=b2)
            t2 = Trainer(forward2, [w2, b2], opt2, checkpointer=ck2)
            ck2.restore_latest()
            resumed = t2.evaluate(val_batch).nelbo
            # loss curve is seamless at the resume point
            self.assertAlmostEqual(float(resumed), saved_loss, places=4)
            t2.fit(build_dataset(data, batch_size=16).repeat(),
                   steps=100)
            final_val = float(t2.evaluate(val_batch).nelbo)
            self.assertLess(final_val, saved_loss)

    def test_early_stopping(self):
        forward, variables, data = self._regression_setup()
        opt = tf.keras.optimizers.Adam(0.05)
        trainer = Trainer(forward, variables, opt)
        val = tf.constant(data[:8], tf.float32)

        def flat_forward(batch):
            # never improves, but keeps a (zero) gradient path so the
            # optimizer has something to apply
            log_lik = tf.constant(1.0)
            for v in variables:
                log_lik = log_lik + tf.reduce_sum(v) * 0.0
            return combine_terms(log_lik, tf.zeros(()), tf.zeros(()))

        trainer.forward = flat_forward
        ds = build_dataset(data, batch_size=8).repeat()
        trainer.fit(ds, steps=500, val_batch=val, log_every=10,
                    early_stop="nelbo", patience=3)
        # stops well before 500 steps (patience 3 -> ~40 steps)
        self.assertLess(trainer.history["step"][-1], 100)
        self.assertTrue(len(trainer.history["val"]) >= 3)


if __name__ == "__main__":
    unittest.main()
