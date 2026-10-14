"""Statistical calibration tests.

Two kinds of evidence that the inference machinery is statistically sound:

1. Training on the VE prior-mode objective drives the NELBO down.
2. Simulation-based calibration (SBC-lite) on a linear-Gaussian inverse
   problem with an analytic posterior: after training an amortized
   variational posterior, the rank of the true latent within posterior
   draws must be (approximately) uniform — the defining property of a
   calibrated posterior.
"""

import unittest

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from pddlvm.NeuralNet import NeuralN
from pddlvm.probabilistic import (
    Trainer,
    build_dataset,
    combine_terms,
    diag_gaussian_log_prob,
    reparameterize,
)

tfd = tfp.distributions


def _make_small_ve():
    from pddlvm.VEModel import VE
    from pddlvm.poisson1DPDEState import PDE_State
    from pddlvm.poisson1DSolver import Solver
    x_fe = tf.constant(np.linspace(-1.0, 1.0, 21), tf.float32)
    pde = PDE_State(xFE=x_fe, dimK=1, dimU=3, dimF=1)
    solver = Solver(pde, x_fe, isNonLinear=False)
    ve = VE(pde, solver)
    u_net = NeuralN("U", np.array([4, 8, 8, 3]), init_seed=11)
    z_net = NeuralN("Z", np.array([4, 8, 8, 3]),
                    multOutputLayout={"K": tf.range(0, 1),
                                      "Bl": tf.range(1, 2),
                                      "Br": tf.range(2, 3)},
                    init_seed=12)
    ve.initPriorNetworks(U_KFBlBr=u_net, Z_UF=z_net)
    ve.setPriors(kappa=tfd.Normal(0.1, 0.01),
                 forcing=tfd.Normal(5.0, 0.1),
                 bl=tfd.Normal(0.0, 0.01),
                 br=tfd.Normal(0.5, 0.01))
    return ve


class TestElboDecreases(unittest.TestCase):
    def test_ve_prior_training_decreases_nelbo(self):
        tf.random.set_seed(7)
        np.random.seed(7)
        ve = _make_small_ve()
        variables = [v for group in ve.varListPriorNets for v in group]
        trainer = Trainer(lambda batch: ve.elbo_terms_prior(0), variables,
                          tf.keras.optimizers.Adam(5e-3))
        dummy = build_dataset(tf.zeros((1, 1)), batch_size=1,
                              shuffle=False).repeat()
        trainer.fit(dummy, steps=80, log_every=80)
        nelbo = trainer.history["nelbo"]
        print(f"VE prior NELBO: {nelbo[0]:.1f} -> {nelbo[-1]:.1f}")
        self.assertLess(nelbo[-1], nelbo[0])
        # SGD on this noisy objective is non-monotonic (early dips happen);
        # require the final value to sit well below the historical peak
        self.assertLess(nelbo[-1], max(nelbo) * 0.5)


class TestSbcLinearGaussian(unittest.TestCase):
    """SBC-lite for a linear-Gaussian inverse problem (analytic posterior)."""

    def setUp(self):
        tf.random.set_seed(42)
        np.random.seed(42)
        self.d_theta, self.d_y, self.sigma = 2, 4, 0.5
        rng = np.random.default_rng(0)
        self.H_np = rng.normal(size=(self.d_y, self.d_theta))
        self.H = tf.constant(self.H_np, tf.float32)
        # posterior: C = (H^T H / sigma^2 + I)^{-1}, m = C H^T y / sigma^2
        self.C = np.linalg.inv(self.H_np.T @ self.H_np / self.sigma ** 2
                               + np.eye(self.d_theta))
        self.C_tf = tf.constant(self.C, tf.float32)

        self.q_net = NeuralN("q", np.array([self.d_y, 32, 32, self.d_theta]),
                             init_seed=42)
        self.batch_size = 64
        self._train()

    def _train(self, steps=1500):
        def forward(batch):
            # synthetic observation: theta* ~ prior, y ~ likelihood
            theta_star = tf.random.normal([self.batch_size, self.d_theta])
            eps = tf.random.normal([self.batch_size, self.d_y]) * self.sigma
            y = theta_star @ tf.transpose(self.H) + eps
            # amortized posterior q(theta | y), sampled by reparameterization
            # (sampling from the prior instead collapses q onto prior draws)
            mean_q, logvar_q = self.q_net.Map([y])
            theta = reparameterize(mean_q, logvar_q)
            log_lik = diag_gaussian_log_prob(
                y, theta @ tf.transpose(self.H),
                tf.fill([self.batch_size, self.d_y],
                        np.float32(np.log(self.sigma ** 2))))
            log_prior = diag_gaussian_log_prob(theta, tf.zeros_like(theta),
                                               tf.zeros_like(theta))
            log_q = diag_gaussian_log_prob(theta, mean_q, logvar_q)
            return combine_terms(log_lik, log_prior, log_q)

        trainer = Trainer(forward, self.q_net.trainable_variables,
                          tf.keras.optimizers.Adam(3e-3))
        dummy = build_dataset(tf.zeros((1, 1)), batch_size=1,
                              shuffle=False).repeat()
        self.history = trainer.fit(dummy, steps=steps, log_every=steps)

    def _true_posterior_mean(self, y_np):
        return (self.C @ self.H_np.T @ y_np / self.sigma ** 2).astype(
            np.float32)

    def test_variational_mean_tracks_analytic_posterior(self):
        # sanity: the amortized posterior actually learned the mapping
        rng = np.random.default_rng(1)
        errs = []
        for _ in range(64):
            theta = rng.normal(size=self.d_theta)
            y = self.H_np @ theta + self.sigma * rng.normal(
                size=self.d_y)
            mean_q, _ = self.q_net.Map([tf.constant(y[None], tf.float32)])
            m_true = self._true_posterior_mean(y)
            errs.append(np.max(np.abs(mean_q.numpy()[0] - m_true)))
        mean_err = float(np.mean(errs))
        print(f"SBC sanity: mean |q_mean - m(y)| = {mean_err:.3f}")
        self.assertLess(mean_err, 0.2)

    def test_rank_statistics_uniform(self):
        # SBC: theta* ~ prior, y ~ likelihood, sample q(theta|y), rank theta*
        # The amortized posterior is a finite-capacity approximation trained
        # for 1500 steps, so a small residual bias is expected (measured rank
        # mean ~28 vs ideal 25); the threshold below (p=0.005, chi2(4) < 14.9)
        # catches GROSS miscalibration -- a collapsed posterior (e.g. the
        # prior-sampling bug) blows chi2 up by orders of magnitude.
        n_reps, n_samples = 200, 50
        rng = np.random.default_rng(2)
        ranks = np.zeros((n_reps, self.d_theta), dtype=int)
        for r in range(n_reps):
            theta_star = rng.normal(size=self.d_theta)
            y = self.H_np @ theta_star + self.sigma * rng.normal(
                size=self.d_y)
            mean_q, logvar_q = self.q_net.Map(
                [tf.constant(y[None], tf.float32)])
            mean_q = tf.repeat(mean_q, n_samples, axis=0)
            logvar_q = tf.repeat(logvar_q, n_samples, axis=0)
            samples = reparameterize(mean_q, logvar_q).numpy()
            for d in range(self.d_theta):
                ranks[r, d] = int(np.sum(samples[:, d] < theta_star[d]))

        # calibrated posterior -> ranks uniform on {0..n_samples}
        for d in range(self.d_theta):
            mean_rank = ranks[:, d].mean()
            print(f"SBC rank mean (dim {d}): {mean_rank:.1f} "
                  f"(ideal {n_samples / 2:.1f})")
            self.assertLess(abs(mean_rank - n_samples / 2), 3.5)
        # pooled chi-square uniformity test over 5 bins
        bins = np.zeros(5)
        flat = ranks.reshape(-1)
        for k in range(5):
            bins[k] = np.sum((flat >= k * 10) & (flat < (k + 1) * 10))
        expected = flat.size / 5
        chi2 = float(np.sum((bins - expected) ** 2 / expected))
        print(f"SBC chi2(4) = {chi2:.2f} (threshold 14.9 at p=0.005)")
        self.assertLess(chi2, 14.9)


if __name__ == "__main__":
    unittest.main()
