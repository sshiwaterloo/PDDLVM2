"""Probabilistic layer for PDDLVM.

- ``distributions`` : diagonal Gaussian log-probs, reparameterization,
  Gaussian residual likelihoods
- ``elbo``          : per-component ELBO records and the physics likelihood
- ``trainer``       : objective-agnostic training loop with metrics,
  learning-rate schedules, early stopping and checkpoints
- ``checkpointing`` : tf.train.CheckpointManager wrapper
"""

from pddlvm.probabilistic.distributions import (  # noqa: F401
    diag_gaussian_log_prob,
    gaussian_residual_log_likelihood,
    reparameterize,
)
from pddlvm.probabilistic.elbo import (  # noqa: F401
    ElboTerms,
    combine_terms,
    physics_gaussian_log_likelihood,
    physics_log_likelihood_batched,
)
from pddlvm.probabilistic.trainer import Trainer, build_dataset  # noqa: F401
from pddlvm.probabilistic.checkpointing import Checkpointer  # noqa: F401

__all__ = [
    "diag_gaussian_log_prob",
    "gaussian_residual_log_likelihood",
    "reparameterize",
    "ElboTerms",
    "combine_terms",
    "physics_gaussian_log_likelihood",
    "physics_log_likelihood_batched",
    "Trainer",
    "build_dataset",
    "Checkpointer",
]
