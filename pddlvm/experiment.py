"""Typed experiment configuration and a reproducible experiment runner.

Design contract
---------------
- Configuration is plain dataclasses serialized to JSON (stdlib only, no
  new dependencies).  Every magic number lives in a config file, never in
  code.
- ``Experiment`` runs the full pipeline — seed -> model build -> training
  -> evaluation -> artifacts — without any interactive call: figures are
  saved to a versioned run directory (matplotlib Agg), never ``show()``'d.
- Artifacts per run: ``metrics.json`` (config snapshot, git sha, library
  versions, per-component history), ``loss.png``, ``samples.png`` and a
  checkpoint directory.  Two runs with the same seed and config reproduce
  the same metrics (verified in ``pddlvm.tests.test_experiment``).
"""

import dataclasses
import datetime
import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")  # must precede pyplot import: pipelines never pop up
from matplotlib import pyplot as plt  # noqa: E402

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
import tensorflow_probability as tfp  # noqa: E402

from pddlvm.NeuralNet import NeuralN  # noqa: E402
from pddlvm.poisson1DPDEState import PDE_State  # noqa: E402
from pddlvm.poisson1DSolver import Solver  # noqa: E402
from pddlvm.VEModel import VE  # noqa: E402
from pddlvm.probabilistic import Checkpointer, Trainer, build_dataset  # noqa: E402

_logger = logging.getLogger(__name__)
tfd = tfp.distributions


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PhysicsConfig:
    dimK: int = 1
    dimU: int = 3
    dimF: int = 1
    num_felems: int = 20
    domain_lo: float = -1.0
    domain_hi: float = 1.0
    epsilon_r: float = 1e-2


@dataclass
class PriorSpec:
    mu: float = 0.0
    sigma: float = 1.0


@dataclass
class PriorsConfig:
    kappa: PriorSpec = field(default_factory=PriorSpec)
    forcing: PriorSpec = field(default_factory=PriorSpec)
    bl: PriorSpec = field(default_factory=PriorSpec)
    br: PriorSpec = field(default_factory=PriorSpec)


@dataclass
class ModelConfig:
    prior_hidden: List[int] = field(default_factory=lambda: [20, 20, 20])


@dataclass
class TrainConfig:
    lr: float = 5e-3
    steps: int = 200
    batch_size: int = 4
    checkpoint_every: int = 100
    early_stop: Optional[str] = None  # None | "nelbo" | "elbo"


@dataclass
class ExperimentConfig:
    name: str = "ve_prior"
    seed: int = 2025
    run_dir: str = "artifacts"
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    priors: PriorsConfig = field(default_factory=PriorsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainConfig = field(default_factory=TrainConfig)

    # ---- (de)serialization (stdlib JSON) ----

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2, default=str)

    @classmethod
    def from_json(cls, payload):
        d = json.loads(payload) if isinstance(payload, str) else payload
        return cls(
            name=d.get("name", "ve_prior"),
            seed=int(d.get("seed", 2025)),
            run_dir=d.get("run_dir", "artifacts"),
            physics=PhysicsConfig(**d.get("physics", {})),
            priors=PriorsConfig(**{
                k: PriorSpec(**v) for k, v in d.get("priors", {}).items()}),
            model=ModelConfig(**d.get("model", {})),
            training=TrainConfig(**d.get("training", {})),
        )

    @classmethod
    def from_file(cls, path):
        with open(path, "r") as fh:
            return cls.from_json(fh.read())


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5)
        return out.stdout.strip()
    except Exception:  # pragma: no cover - environment-dependent
        return "unknown"


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

class Experiment:
    """A full prior-mode PDDLVM run: build, train, evaluate, save artifacts."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._apply_seed(config.seed)

        phy = config.physics
        x_fe = tf.constant(np.linspace(phy.domain_lo, phy.domain_hi,
                                       phy.num_felems + 1), tf.float32)
        pde = PDE_State(xFE=x_fe, dimK=phy.dimK, dimU=phy.dimU, dimF=phy.dimF)
        solver = Solver(pde, x_fe, isNonLinear=False)

        # prior networks with configurable widths and deterministic
        # per-layer init seeds (required for reproducible experiments)
        hidden = config.model.prior_hidden
        dim_kfblbr = phy.dimK + phy.dimF + 2
        u_net = NeuralN("U_of_KFBlBr",
                        np.array([dim_kfblbr] + list(hidden) + [phy.dimU]),
                        init_seed=config.seed)
        z_net = NeuralN("Z_of_UF",
                        np.array([phy.dimU + phy.dimF] + list(hidden)
                                 + [phy.dimK + 2]),
                        multOutputLayout={
                            "K": tf.range(0, phy.dimK),
                            "Bl": tf.range(phy.dimK, phy.dimK + 1),
                            "Br": tf.range(phy.dimK + 1, phy.dimK + 2)},
                        init_seed=config.seed)
        self.ve = VE(pde, solver)
        self.ve.initPriorNetworks(U_KFBlBr=u_net, Z_UF=z_net)
        self.ve.setPriors(
            kappa=tfd.Normal(config.priors.kappa.mu,
                             config.priors.kappa.sigma),
            forcing=tfd.Normal(config.priors.forcing.mu,
                               config.priors.forcing.sigma),
            bl=tfd.Normal(config.priors.bl.mu, config.priors.bl.sigma),
            br=tfd.Normal(config.priors.br.mu, config.priors.br.sigma))

        self.variables = [v for group in self.ve.varListPriorNets
                          for v in group]
        self.optimizer = tf.keras.optimizers.Adam(config.training.lr)

        def forward(batch):
            # prior-mode ELBO is batch-independent (latents are sampled
            # from the priors); the dataset just drives the step count
            return self.ve.elbo_terms_prior(0)

        self.forward = forward

    @staticmethod
    def _apply_seed(seed):
        tf.random.set_seed(seed)
        np.random.seed(seed)

    def _make_run_dir(self):
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = os.path.join(
            self.config.run_dir,
            f"{self.config.name}_seed{self.config.seed}_{stamp}")
        os.makedirs(os.path.join(run_dir, "ckpts"), exist_ok=True)
        return run_dir

    def run(self, steps=None, seed=None):
        """Train and write artifacts; returns (run_dir, metrics dict)."""
        if seed is not None:
            self._apply_seed(seed)
        tr = self.config.training
        steps = tr.steps if steps is None else steps

        run_dir = self._make_run_dir()
        checkpointer = Checkpointer(os.path.join(run_dir, "ckpts"),
                                    optimizer=self.optimizer,
                                    **{f"v{i}": v
                                       for i, v in enumerate(self.variables)})
        trainer = Trainer(self.forward, self.variables, self.optimizer,
                          checkpointer=checkpointer,
                          checkpoint_every=tr.checkpoint_every)
        dummy = build_dataset(tf.zeros((max(tr.batch_size, 1), 1)),
                              batch_size=max(tr.batch_size, 1),
                              shuffle=False).repeat()
        history = trainer.fit(dummy, steps=steps, log_every=steps)

        metrics = self._write_metrics(run_dir, history, steps)
        self._plot_loss(run_dir, history)
        self._plot_samples(run_dir)
        _logger.info("experiment done: %s", run_dir)
        return run_dir, metrics

    def _write_metrics(self, run_dir, history, steps):
        final = {name: history[name][-1] for name in
                 ("log_likelihood", "log_prior", "log_q", "elbo", "nelbo")}
        metrics = {
            "config": dataclasses.asdict(self.config),
            "meta": {
                "git_sha": _git_sha(),
                "tf_version": tf.__version__,
                "tfp_version": tfp.__version__,
                "python_version": __import__("platform").python_version(),
                "steps": steps,
                "timestamp": datetime.datetime.now().isoformat(),
            },
            "final": final,
            "history": {k: history[k] for k in
                        ("step", "log_likelihood", "log_prior", "log_q",
                         "elbo", "nelbo", "lr")},
        }
        with open(os.path.join(run_dir, "metrics.json"), "w") as fh:
            json.dump(metrics, fh, indent=2)
        return metrics

    def _plot_loss(self, run_dir, history):
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, label in (("nelbo", "NELBO"), ("elbo", "ELBO"),
                            ("log_likelihood", "log likelihood"),
                            ("log_prior", "log prior"), ("log_q", "log q")):
            ax.plot(history["step"], history[name], label=label)
        ax.set_xlabel("step")
        ax.set_ylabel("value")
        ax.set_title(f"{self.config.name} (seed {self.config.seed})")
        ax.legend()
        ax.grid(True)
        fig.savefig(os.path.join(run_dir, "loss.png"), dpi=150,
                    bbox_inches="tight")
        plt.close(fig)

    def _plot_samples(self, run_dir, n_samples=5):
        fig, ax = plt.subplots(figsize=(8, 5))
        x = self.ve.pdeState.xFE.numpy()
        for _ in range(n_samples):
            f_i = self.ve.LatentForcingPrior.sample(sample_shape=(1))
            k_i = self.ve.LatentKappaPrior.sample(sample_shape=(1))
            bl = self.ve.LatentBlPrior.sample(sample_shape=(1))
            br = self.ve.LatentBrPrior.sample(sample_shape=(1))
            f_i = tf.reshape(f_i, [1, -1])
            k_i = tf.reshape(k_i, [1, -1])
            bl = tf.reshape(bl, [1, -1])
            br = tf.reshape(br, [1, -1])
            mean_u, _ = self.ve.U_KFBlBr.Map(
                [k_i, f_i, bl, br])
            u = self.ve.pdeState.u_funXFE(
                tf.reshape(mean_u, [-1, 1])).numpy()
            ax.plot(x, u, alpha=0.6, lw=1)
        ax.set_xlabel("x")
        ax.set_ylabel("u")
        ax.set_title(f"{self.config.name}: sampled fields (prior mode)")
        fig.savefig(os.path.join(run_dir, "samples.png"), dpi=150,
                    bbox_inches="tight")
        plt.close(fig)
