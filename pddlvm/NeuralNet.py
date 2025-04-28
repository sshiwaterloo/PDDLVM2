# NeuralNet.py
# ------------
import tensorflow as tf
from tensorflow import keras
import numpy as np

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

    def __init__(self, name, layout, transf=None, multOutputLayout=None):
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
        """
        super(NeuralN, self).__init__()
        self.NN_name = name
        self.layout = layout
        self.multOutputLayout = multOutputLayout

        self.activation = 'swish'
        self.initializer = tf.keras.initializers.HeNormal()

        self.lower_alpha = tf.constant(1e-6, dtype=tf.float32)
        self.upper_w = tf.constant(1.0, dtype=tf.float32)

        self.hiddenLayerType = keras.layers.Dense

        if transf is None:
            def identity(a):
                return a
            self.transf_forward = identity
        else:
            self.transf_forward = transf

        self.initialize()

    def initialize(self):
        """
        Builds the internal Keras Sequential model according to `layout`.
        The final Dense layer outputs 2 * (layout[-1]) units, split into (mean, logvar).
        Prints a summary of the constructed layers.
        """
        self.NN = keras.Sequential(name=self.NN_name)

        # Build hidden layers based on layout
        for i in range(1, self.layout.shape[0] - 1):
            if i == 1:
                print('first layer')
                self.NN.add(
                    self.hiddenLayerType(
                        self.layout[i],
                        activation=self.activation,
                        kernel_initializer=self.initializer
                    )
                )
            else:
                print('inner layer ', i)
                self.NN.add(
                    self.hiddenLayerType(
                        self.layout[i],
                        activation=self.activation,
                        kernel_initializer=self.initializer
                    )
                )

        print('last layer')
        # Final layer => produce 2 * (layout[-1]) => (mean, logvar)
        self.NN.add(
            keras.layers.Dense(self.layout[-1] * 2)
        )

        # Explicitly build with a known input shape => (None, layout[0]) for summary
        input_dim = self.layout[0]
        self.NN.build((None, input_dim))

        # Print model summary
        self.NN.summary()

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