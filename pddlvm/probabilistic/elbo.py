"""ELBO objectives with per-component accounting.

The two PDDLVM objectives decompose as

    prior mode:     elbo = mean( log_likelihood + log_prior  - log_q )
    posterior mode: elbo = mean( log_likelihood + log_prior  - log_q )

with different meanings of the terms:

- prior mode (physics prior): ``log_likelihood`` is the Gaussian physics
  residual of the sampled field u under the PDE operator; ``log_prior`` is
  the prior density of the latent Z given (u, f) through the beta network;
  ``log_q`` is the density of u under the alpha network.
- posterior mode (data inference): ``log_likelihood`` is the observation
  likelihood; ``log_prior`` collects the latent priors (Z | theta network
  and the configured priors on kappa / boundary values); ``log_q`` collects
  the variational posterior densities.

Every term is returned separately so that training logs can record each
component.
"""

from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from pddlvm.probabilistic.distributions import LOG_2PI


@dataclass
class ElboTerms:
    """One training step's ELBO broken into components.

    All fields are per-step (reduced) scalar tensors except where noted.
    """

    log_likelihood: tf.Tensor
    log_prior: tf.Tensor
    log_q: tf.Tensor
    elbo: tf.Tensor
    nelbo: tf.Tensor

    def as_dict(self):
        """Float values for logging (numpy scalars)."""
        out = {}
        for name in ("log_likelihood", "log_prior", "log_q", "elbo", "nelbo"):
            out[name] = float(getattr(self, name))
        return out


def combine_terms(log_likelihood, log_prior, log_q,
                  reduce=tf.reduce_mean):
    """elbo = reduce(log_likelihood + log_prior - log_q)."""
    elbo = reduce(log_likelihood + log_prior - log_q)
    return ElboTerms(
        log_likelihood=reduce(log_likelihood),
        log_prior=reduce(log_prior),
        log_q=reduce(log_q),
        elbo=elbo,
        nelbo=-elbo,
    )


def physics_gaussian_log_likelihood(a_mat, f_vec, u, sigma):
    """Scalar log N(A u - f; 0, sigma^2 I) for one sample u.

    Matches the legacy ``VEModel.physicsPenalty`` formula exactly:
        -0.5 * (N * log(sigma^2) + ||A u - f||^2 / sigma^2 + log 2 pi)
    where N is the system size.
    """
    sigma = tf.convert_to_tensor(sigma, dtype=u.dtype)
    d = a_mat @ u[:, None] - f_vec[:, None]
    n = tf.cast(tf.shape(a_mat)[0], d.dtype)
    return -0.5 * (
        n * tf.math.log(sigma ** 2)
        + tf.squeeze(tf.linalg.norm(d, axis=0) ** 2) / sigma ** 2
        + LOG_2PI
    )


def physics_log_likelihood_batched(a_mat, f_vec, u_batch, sigma):
    """log N(A u - f; 0, sigma^2 I) for a batch u_batch of shape [B, N]."""
    sigma = tf.convert_to_tensor(sigma, dtype=u_batch.dtype)
    d = tf.transpose(u_batch @ tf.transpose(a_mat)) - f_vec  # [B, N]
    n = tf.cast(tf.shape(a_mat)[0], d.dtype)
    return -0.5 * (
        n * tf.math.log(sigma ** 2)
        + tf.reduce_sum(d ** 2, axis=-1) / sigma ** 2
        + LOG_2PI
    )
