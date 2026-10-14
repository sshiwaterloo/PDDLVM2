"""Tests for the experiment system.

Covers: JSON config round-trip; a full non-interactive run producing all
artifacts (metrics.json / loss.png / samples.png / checkpoints); and
reproducibility — two runs with the same seed yield the same training
history.
"""

import json
import os
import tempfile
import unittest

from pddlvm.experiment import Experiment, ExperimentConfig


def _small_config(run_dir, seed=2025, steps=30):
    cfg = ExperimentConfig()
    cfg.name = "test_ve_prior"
    cfg.seed = seed
    cfg.run_dir = run_dir
    cfg.model.prior_hidden = [8, 8]
    cfg.physics.num_felems = 20
    cfg.training.steps = steps
    cfg.training.checkpoint_every = 15
    return cfg


class TestConfig(unittest.TestCase):
    def test_json_roundtrip(self):
        cfg = _small_config("unused")
        cfg.priors.kappa.mu = 0.25
        cfg2 = ExperimentConfig.from_json(cfg.to_json())
        self.assertEqual(cfg2, cfg)
        self.assertEqual(cfg2.priors.kappa.mu, 0.25)
        self.assertEqual(cfg2.model.prior_hidden, [8, 8])

    def test_from_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cfg.json")
            with open(path, "w") as fh:
                fh.write(_small_config("x", seed=7).to_json())
            cfg = ExperimentConfig.from_file(path)
            self.assertEqual(cfg.seed, 7)


class TestExperiment(unittest.TestCase):
    def test_run_produces_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp = Experiment(_small_config(tmp, steps=30))
            run_dir, metrics = exp.run()
            self.assertTrue(os.path.isdir(run_dir))
            for fname in ("metrics.json", "loss.png", "samples.png"):
                self.assertTrue(os.path.isfile(
                    os.path.join(run_dir, fname)), fname)
            self.assertTrue(os.path.isdir(os.path.join(run_dir, "ckpts")))
            self.assertTrue(os.listdir(os.path.join(run_dir, "ckpts")))

            with open(os.path.join(run_dir, "metrics.json")) as fh:
                metrics = json.load(fh)
            self.assertEqual(metrics["meta"]["steps"], 30)
            self.assertEqual(len(metrics["history"]["step"]), 30)
            for key in ("log_likelihood", "log_prior", "log_q",
                        "elbo", "nelbo"):
                self.assertEqual(len(metrics["history"][key]), 30)
                self.assertTrue(all(
                    isinstance(v, float) for v in metrics["history"][key]))
            self.assertIn("config", metrics)
            self.assertIn("git_sha", metrics["meta"])

    def test_reproducibility_same_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            run1, m1 = Experiment(_small_config(tmp, seed=99, steps=25)).run()
            run2, m2 = Experiment(_small_config(tmp, seed=99, steps=25)).run()
            h1 = m1["history"]
            h2 = m2["history"]
            self.assertEqual(h1["step"], h2["step"])
            for key in ("log_likelihood", "log_prior", "log_q",
                        "elbo", "nelbo"):
                for a, b in zip(h1[key], h2[key]):
                    self.assertAlmostEqual(a, b, places=5,
                                           msg=f"{key} differs: {a} vs {b}")
            self.assertAlmostEqual(m1["final"]["nelbo"],
                                   m2["final"]["nelbo"], places=5)


if __name__ == "__main__":
    unittest.main()
