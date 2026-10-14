# NeuralNet.py
# ------------
import logging

import tensorflow as tf
from tensorflow import keras
import numpy as np

_logger = logging.getLogger(__name__)

class ConvFieldNet(tf.keras.Model):
    """Conv3D-transpose decoder: latent z -> 3D field with (mean, logvar).

    A dense layer maps z to a coarse (2, 2, 2) cube of filters, which
    ``Conv3DTranspose`` blocks (stride 2, 'same') upsample to
    ``field_shape``, and a final Conv3D head outputs 2 * channels values
    split into (mean, logvar) — the same output convention as ``NeuralN``.
    Dense decoders are the wrong tool for 3D fields; this is the
    decoder used for the 3D Navier-Stokes demos.

    Parameters
    ----------
    name : str
    z_dim : int
    field_shape : (n, n, n) with n a power of two
    channels : output field channels (1 scalar field, 3 for u, v, w)
    base_filters : filters at the finest level (doubled per block upward)
    init_seed : per-layer deterministic init seed (see NeuralN)
    """

    def __init__(self, name, z_dim, field_shape, channels=1, base_filters=16,
                 init_seed=None, verbose=False):
        super(ConvFieldNet, self).__init__()
        self.NN_name = name
        self.field_shape = tuple(int(s) for s in field_shape)
        self.channels = int(channels)
        self.verbose = verbose
        self.init_seed = init_seed
        n = self.field_shape[0]
        if n & (n - 1):
            raise ValueError("field_shape[0] must be a power of two")
        self.levels = int(np.log2(n)) - 1  # 2 -> n via stride-2 blocks
        self.start_filters = int(base_filters) * (2 ** self.levels)

        def initializer(layer_index):
            if init_seed is None:
                return tf.keras.initializers.HeNormal()
            return tf.keras.initializers.HeNormal(
                seed=int(init_seed) + 1000 * int(layer_index))

        self.dense = keras.layers.Dense(
            8 * self.start_filters, kernel_initializer=initializer(0),
            name=f"{name}_dense")
        self.blocks = []
        filters = self.start_filters
        for i in range(self.levels):
            filters = max(filters // 2, int(base_filters))
            self.blocks.append(keras.layers.Conv3DTranspose(
                filters, kernel_size=3, strides=2, padding="same",
                activation="swish", kernel_initializer=initializer(1 + i)))
        self.head = keras.layers.Conv3D(
            2 * self.channels, kernel_size=3, padding="same",
            kernel_initializer=initializer(1 + self.levels))

    def Map(self, z):
        """z ([batch, z_dim] or a list thereof) -> (mean, logvar), each
        [batch, n, n, n, channels]."""
        if not isinstance(z, (list, tuple)):
            z = [z]
        flat = tf.concat([tf.reshape(v, [tf.shape(v)[0], -1]) for v in z],
                         axis=1)
        h = self.dense(flat)
        h = tf.reshape(h, [-1, 2, 2, 2, self.start_filters])
        for blk in self.blocks:
            h = blk(h)
        out = self.head(h)
        mean = out[..., :self.channels]
        logvar = out[..., self.channels:]
        return mean, logvar


class ConvFieldNet2D(tf.keras.Model):
    """Conv2D-transpose decoder: latent z -> 2D field with (mean, logvar).

    The 2D analogue of ``ConvFieldNet`` (used for the cylinder-flow demo).  A
    dense layer maps z to a coarse (2, 2) patch of filters, which
    ``Conv2DTranspose`` blocks (stride 2, 'same') upsample to ``field_shape``,
    and a final Conv2D head outputs 2 * channels values split into
    (mean, logvar) -- the same output convention as ``NeuralN``/``ConvFieldNet``.
    """

    def __init__(self, name, z_dim, field_shape, channels=1, base_filters=16,
                 init_seed=None, verbose=False):
        super(ConvFieldNet2D, self).__init__()
        self.NN_name = name
        self.field_shape = tuple(int(s) for s in field_shape)
        self.channels = int(channels)
        self.verbose = verbose
        self.init_seed = init_seed
        n = self.field_shape[0]
        if n & (n - 1):
            raise ValueError("field_shape[0] must be a power of two")
        self.levels = int(np.log2(n)) - 1  # 2 -> n via stride-2 blocks
        self.start_filters = int(base_filters) * (2 ** self.levels)

        def initializer(layer_index):
            if init_seed is None:
                return tf.keras.initializers.HeNormal()
            return tf.keras.initializers.HeNormal(
                seed=int(init_seed) + 1000 * int(layer_index))

        self.dense = keras.layers.Dense(
            4 * self.start_filters, kernel_initializer=initializer(0),
            name=f"{name}_dense")
        self.blocks = []
        filters = self.start_filters
        for i in range(self.levels):
            filters = max(filters // 2, int(base_filters))
            self.blocks.append(keras.layers.Conv2DTranspose(
                filters, kernel_size=3, strides=2, padding="same",
                activation="swish", kernel_initializer=initializer(1 + i)))
        self.head = keras.layers.Conv2D(
            2 * self.channels, kernel_size=3, padding="same",
            kernel_initializer=initializer(1 + self.levels))

    def Map(self, z):
        """z ([batch, z_dim] or a list thereof) -> (mean, logvar), each
        [batch, ny, nx, channels]."""
        if not isinstance(z, (list, tuple)):
            z = [z]
        flat = tf.concat([tf.reshape(v, [tf.shape(v)[0], -1]) for v in z],
                         axis=1)
        h = self.dense(flat)
        h = tf.reshape(h, [-1, 2, 2, self.start_filters])
        for blk in self.blocks:
            h = blk(h)
        out = self.head(h)
        mean = out[..., :self.channels]
        logvar = out[..., self.channels:]
        return mean, logvar


class NeuralN(tf.keras.Model):
    """
    Implements a neural network that generates a mean and log-variance output
    from an input tensor. The final layer produces 2 * (output_dim) values
    split into (mean, logvar).

    Attributes
    ----------
    NN_name : str
        The name of this neural network instance.
    layout : np.array
        Sequence defining input_dim, hidden-layer widths, and final output_dim.
        The final integer in `layout` is the dimensionality of the mean/logvar.
    multOutputLayout : Any
        Placeholder for future multi-output expansions; not utilized.
    activation : str
        Name of the activation function used in hidden layers.
    initializer : tf.keras.initializers.Initializer
        TensorFlow initializer object for layer weights.
    lower_alpha : tf.Tensor
        Lower bound for variance adjustments (float32).
    upper_w : tf.Tensor
        Upper bound for variance adjustments (float32).
    hiddenLayerType : tf.keras.layers.Layer
        Layer type for hidden layers (Dense by default).
    transf_forward : Callable
        Transformation applied to input before forwarding through layers.
    NN : tf.keras.Sequential
        Internal Keras Sequential model that defines the layer structure.

    Notes
    -----
    The layout array typically looks like:
      [input_dim, hidden1, hidden2, ..., output_dim]
    The actual final Dense layer will generate 2 * output_dim values,
    which are interpreted as (mean, logvar).
    """

    def __init__(self, name, layout, transf=None, multOutputLayout=None,
                 verbose=False, init_seed=None):
        """
        Initializes the NeuralN class.

        Parameters
        ----------
        name : str
            Name of the network instance.
        layout : np.array
            Defines the layer dimensions: [input_dim, hidden1, ..., output_dim].
            The final output_dim corresponds to half the final layer's actual
            size, because mean/logvar are each of size output_dim.
        transf : Callable or None
            Function applied to input. If None, the identity transform is used.
        multOutputLayout : Any
            Placeholder for multi-output expansions; not utilized.
        verbose : bool
            If True, print the Keras layer summary at construction time.
        init_seed : int or None
            Base seed for weight initialization.  When set, each layer gets
            a deterministic per-layer seed (base + offset), so two networks
            built with the same ``init_seed`` have identical initial
            weights -- required for reproducible experiments (keras
            initializers without an explicit seed are NOT reproducible
            across constructions, even after tf.random.set_seed).
        """
        super(NeuralN, self).__init__()
        self.NN_name = name
        self.layout = layout
        self.multOutputLayout = multOutputLayout

        self.activation = 'swish'
        self.initializer = tf.keras.initializers.HeNormal()
        self.init_seed = init_seed

        self.lower_alpha = tf.constant(1e-6, dtype=tf.float32)
        self.upper_w = tf.constant(1.0, dtype=tf.float32)

        self.hiddenLayerType = keras.layers.Dense

        if transf is None:
            def identity(a):
                return a
            self.transf_forward = identity
        else:
            self.transf_forward = transf

        self.verbose = verbose
        self.initialize()

    def initialize(self):
        """
        Builds the internal Keras Sequential model according to `layout`.
        The final Dense layer outputs 2 * (layout[-1]) units, split into (mean, logvar).
        Prints a summary of the constructed layers when ``verbose`` is set.
        """
        self.NN = keras.Sequential(name=self.NN_name)

        # Build hidden layers based on layout
        for i in range(1, self.layout.shape[0] - 1):
            _logger.debug('adding hidden layer %d of %s', i, self.NN_name)
            initializer = self._layer_initializer(i)
            self.NN.add(
                self.hiddenLayerType(
                    int(self.layout[i]),
                    activation=self.activation,
                    kernel_initializer=initializer
                )
            )

        _logger.debug('adding final layer of %s', self.NN_name)
        # Final layer => produce 2 * (layout[-1]) => (mean, logvar)
        self.NN.add(
            keras.layers.Dense(int(self.layout[-1]) * 2,
                               kernel_initializer=self._layer_initializer(
                                   self.layout.shape[0]))
        )

        # Explicitly build with a known input shape => (None, layout[0]) for summary
        input_dim = int(self.layout[0])
        self.NN.build((None, input_dim))

        # Print model summary
        if self.verbose:
            self.NN.summary()

    def _layer_initializer(self, layer_index):
        """Initializer for a layer: unseeded, or deterministic per-layer
        seed derived from ``init_seed``."""
        if self.init_seed is None:
            return self.initializer
        return tf.keras.initializers.HeNormal(
            seed=int(self.init_seed) + 1000 * int(layer_index))

    def Map(self, varIn):
        """
        Forward pass to compute (mean, logvar).

        Parameters
        ----------
        varIn : tf.Tensor or list of tf.Tensor
            Input(s) of rank=2. All inputs are concatenated along axis=1.

        Returns
        -------
        mean : tf.Tensor
            The mean output of shape (batch_size, layout[-1]).
        logvar : tf.Tensor
            The stabilized log-variance output of shape (batch_size, layout[-1]).

        Raises
        ------
        tf.errors.InvalidArgumentError
            If any of the intermediate or final computed values contain NaN or Inf.
        """
        if not isinstance(varIn, (list, tuple)):
            varIn = [varIn]

        varIn = self.transf_forward(varIn)

        allVarList = []
        for v in varIn:
            tf.debugging.assert_rank(
                v, 2,
                message=f"NeuralN.Map expects rank=2 inputs, got {v.shape}"
            )
            allVarList.append(v)

        # Concatenate all inputs along axis=1 => shape=(batch_size, input_dim)
        allVar = tf.concat(allVarList, axis=1)

        mapped = self.NN(allVar)
        out_dim = self.layout[-1]

        mean, logvar = mapped[:, :out_dim], mapped[:, out_dim:]

        tf.debugging.assert_all_finite(mean, 'Map => mean has NaN/Inf')
        tf.debugging.assert_all_finite(logvar, 'Map => logvar has NaN/Inf')

        logvar = self.dBoundwalpha_log(logvar)
        return mean, logvar

    def dBoundwalpha(self, X):
        """
        Applies a sigmoid-based bounding to squeeze values into [lower_alpha, upper_w].

        Parameters
        ----------
        X : tf.Tensor
            Input tensor.

        Returns
        -------
        tf.Tensor
            Tensor with values in the range [lower_alpha, upper_w].
        """
        return (
            self.lower_alpha +
            (self.upper_w - self.lower_alpha) * tf.nn.sigmoid(X)
        )

    def dBoundwalpha_log(self, X):
        """
        Applies the dBoundwalpha transform, then takes logarithm of the result.

        Parameters
        ----------
        X : tf.Tensor
            Input tensor.

        Returns
        -------
        tf.Tensor
            The log of the bounding transform, ensuring variance is within a safe range.
        """
        return tf.math.log(self.dBoundwalpha(X))