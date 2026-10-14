"""
Unit tests for the NeuralN class using the unittest framework.

The tests include verification of:
1) The shapes of the output mean and log-variance.
2) The absence of NaN or Inf values.
3) Correct concatenation of multiple input tensors.
"""

import unittest
import numpy as np
import tensorflow as tf
from pddlvm.NeuralNet import NeuralN

class TestNeuralN(unittest.TestCase):
    """
    Collection of unit tests for the NeuralN class.

    Attributes
    ----------
    None
    """

    def test_map_shapes(self):
        """
        Checks that NeuralN produces output mean and logvar of the expected shapes.

        A simple layout is used: [3, 10, 4]. The final layer
        outputs 2*(4) = 8 units, corresponding to mean and logvar each of size 4.

        Returns
        -------
        None
            This method does not return anything. It relies on assertions.

        Raises
        ------
        AssertionError
            If the shapes of mean or logvar do not match the expected (5, 4).
        """
        layout = np.array([3, 10, 4])
        model = NeuralN(name="TestNetShapes", layout=layout)

        x = tf.random.normal(shape=(5, 3), dtype=tf.float32)  # (batch=5, input_dim=3)
        mean, logvar = model.Map(x)

        self.assertEqual(mean.shape, (5, 4), "Expected mean shape (5,4)")
        self.assertEqual(logvar.shape, (5, 4), "Expected logvar shape (5,4)")

    def test_no_nan_inf(self):
        """
        Ensures the mean and logvar outputs are finite for typical input ranges.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If either mean or logvar contain NaN or Inf values.
        """
        layout = np.array([2, 8, 2])
        model = NeuralN(name="FiniteCheck", layout=layout)

        x = tf.random.normal(shape=(6, 2), dtype=tf.float32)  # (batch=6, input_dim=2)
        mean, logvar = model.Map(x)

        self.assertTrue(tf.reduce_all(tf.math.is_finite(mean)), "Mean contains NaN/Inf")
        self.assertTrue(tf.reduce_all(tf.math.is_finite(logvar)), "Logvar contains NaN/Inf")

    def test_concat_inputs(self):
        """
        Verifies that providing multiple rank-2 Tensors to Map(...) concatenates them correctly.

        In this example, each tensor is of shape (5, 3), so the final input
        dimension is 6, matching layout[0]. Layout is [6, 8, 3], meaning the
        final layer outputs 2*3=6 units.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If mean or logvar shapes do not match (5,3) after the concatenation.
        """
        layout = np.array([6, 8, 3])
        model = NeuralN(name="ConcatTest", layout=layout)

        x1 = tf.ones((5, 3), dtype=tf.float32)
        x2 = tf.zeros((5, 3), dtype=tf.float32)

        mean, logvar = model.Map([x1, x2])

        self.assertEqual(mean.shape, (5, 3), "Mean shape mismatch for concatenated input")
        self.assertEqual(logvar.shape, (5, 3), "Logvar shape mismatch for concatenated input")


    def test_init_seed_reproducible(self):
        """Same init_seed -> identical initial weights (needed for
        reproducible experiments: unseeded keras initializers are not
        reproducible across constructions)."""
        m1 = NeuralN("s1", np.array([3, 10, 4]), init_seed=7)
        m2 = NeuralN("s2", np.array([3, 10, 4]), init_seed=7)
        self.assertEqual(len(m1.trainable_variables),
                         len(m2.trainable_variables))
        for w1, w2 in zip(m1.trainable_variables, m2.trainable_variables):
            np.testing.assert_array_equal(w1.numpy(), w2.numpy())


if __name__ == "__main__":
    unittest.main()