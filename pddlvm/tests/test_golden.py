"""Golden-value regression tests.

Locks numerical behaviour of the core paths so that silent regressions
(solver drift, objective changes, numerical-format mistakes) fail loudly.
Values are recorded from a known-good commit; tolerances are relative and
deliberately loose enough to absorb last-ulp platform differences while
still catching real regressions.
"""

import tempfile
import unittest

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from pddlvm.experiment import Experiment, ExperimentConfig
from pddlvm.numerics.fd import solve_burgers_transient
from pddlvm.numerics.mms import (
    burgers_mms_source_transient,
    burgers_mms_u_transient,
    l2_error,
)

tfd = tfp.distributions


class TestGoldenValues(unittest.TestCase):
    def test_ve_prior_training_final_nelbo(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = ExperimentConfig()
            cfg.name = "golden"
            cfg.seed = 123
            cfg.run_dir = tmp
            cfg.model.prior_hidden = [8, 8]
            cfg.physics.num_felems = 20
            cfg.training.steps = 40
            cfg.training.checkpoint_every = 100
            _, metrics = Experiment(cfg).run()
            final = metrics["final"]["nelbo"]
            # golden value recorded from a verified run
            self.assertAlmostEqual(final, 7194885.5, delta=7194885.5 * 1e-3,
                                   msg=f"nelbo={final}")

    def test_burgers_transient_mms_error(self):
        nu, t1 = 0.1, 0.5
        x = np.linspace(0.0, 1.0, 129)
        u0 = burgers_mms_u_transient(x, 0.0, nu)
        u, _, _ = solve_burgers_transient(
            x, u0, nu,
            source=lambda xx, t: burgers_mms_source_transient(xx, t, nu),
            t0=0.0, t1=t1, dt=4e-5,
            bc_left=lambda t: burgers_mms_u_transient(0.0, t, nu),
            bc_right=lambda t: burgers_mms_u_transient(1.0, t, nu))
        err = l2_error(u, burgers_mms_u_transient(x, t1, nu), x)
        self.assertAlmostEqual(err, 0.0095609, delta=0.0095609 * 1e-2,
                               msg=f"mms err={err}")

    def test_space_time_residual_mms(self):
        from pddlvm.numerics.fd import burgers_space_time_residual
        nu = 0.1
        x = np.linspace(0.0, 1.0, 65)
        t = np.linspace(0.0, 0.5, 65)
        tt, xx = np.meshgrid(t, x, indexing="ij")
        u = burgers_mms_u_transient(xx, tt, nu)
        r = burgers_space_time_residual(
            u, x, t, nu,
            ic=lambda xv: burgers_mms_u_transient(xv, 0.0, nu),
            bc_left=lambda tv: burgers_mms_u_transient(0.0, tv, nu),
            bc_right=lambda tv: burgers_mms_u_transient(1.0, tv, nu),
            source=lambda xv, tv: burgers_mms_source_transient(xv, tv, nu))
        m = float(np.max(np.abs(r)))
        self.assertAlmostEqual(m, 1.04e-2, delta=1.04e-2 * 1e-2,
                               msg=f"residual max={m}")


if __name__ == "__main__":
    unittest.main()
