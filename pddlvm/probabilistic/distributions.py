"""Distribution utilities for the variational objectives.

Pure functions, no side effects.  The log-probability conventions match the
legacy ``VEModel`` formulas exactly (diagonal Gaussians, summed over the
feature axis; Gaussian residuals for the physics/data likelihood).
"""

import numpy as np
import tensorflow as tf

# python float (not a tensor): promotes to any tensor dtype without a
# float32 round-trip
LOG_2PI = float(np.log(2.0 * np.pi))


def diag_gaussian_log_prob(x, mean, logvar, axis=-1, dtype=None):
    """log N(x; mean, diag(exp(logvar))), summed over ``axis``."""
    if dtype is not None:
        x = tf.cast(x, dtype)
        mean = tf.cast(mean, dtype)
        logvar = tf.cast(logvar, dtype)
    return tf.reduce_sum(
        -0.5 * ((x - mean) ** 2 * tf.exp(-logvar) + logvar + LOG_2PI),
        axis=axis)


def reparameterize(mean, logvar, seed=None):
    """Reparameterized sample z = mean + exp(logvar / 2) * eps, eps ~ N(0, I)."""
    eps = tf.random.normal(shape=tf.shape(mean), dtype=mean.dtype, seed=seed)
    z = eps * tf.exp(logvar * 0.5) + mean
    tf.debugging.assert_all_finite(z, "reparameterize produced non-finite z")
    return z


def gaussian_residual_log_likelihood(residuals, sigma):
    """Per-sample log likelihood of residuals under N(0, sigma^2 I).

    ``residuals`` has shape [batch, M]; the result has shape [batch]:
        -0.5 * (sum(r^2) / sigma^2 + M * log(2 pi sigma^2))
    """
    residuals = tf.convert_to_tensor(residuals)
    sigma = tf.convert_to_tensor(sigma, dtype=residuals.dtype)
    m = tf.cast(tf.shape(residuals)[-1], residuals.dtype)
    sq = tf.reduce_sum(residuals ** 2, axis=-1)
    return -0.5 * (sq / sigma ** 2 + m * tf.math.log(2.0 * np.pi * sigma ** 2))
