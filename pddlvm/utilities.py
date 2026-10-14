"""Shared TensorFlow utilities.

This module intentionally has **no import-time side effects**: it imports
tensorflow / tensorflow_probability / numpy only, and must not touch
matplotlib (plotting style helpers live in ``pddlvm.plotting`` and are
applied explicitly).  See ``pddlvm.tests.test_side_effects`` for the
guards.
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfb = tfp.bijectors
tfm = tf.math
floatf = tf.float32


def updateTfFloat(newFloat):
    """Switch the package-wide float dtype (legacy global; prefer passing
    ``dtype`` explicitly to numerics functions)."""
    global floatf
    floatf = newFloat


def tffloat(x):
    return tf.constant(x, dtype=floatf)


def tfint(x):
    return tf.constant(x, dtype=tf.int32)


# ! Some constants
_0 = tffloat(0.)
_PI = tffloat(np.pi)
