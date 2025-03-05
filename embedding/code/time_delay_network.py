import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.initializers import HeNormal

@tf.keras.utils.register_keras_serializable()
class TDNN(layers.Layer):
    def __init__(self, output_dim, dilation_rate, num_frames_per_filter, **kwargs):
        super(TDNN, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.dilation_rate = dilation_rate
        self.num_frames_per_filter = num_frames_per_filter
        
        # Use a slightly smaller initialization scale
        self.conv = layers.Conv1D(
            filters=self.output_dim,
            kernel_size=self.num_frames_per_filter,
            dilation_rate=self.dilation_rate,
            strides=1, 
            padding='valid', 
            activation='relu',
            kernel_initializer=HeNormal(seed=42),
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)  # Add L2 regularization
        )

    def call(self, inputs):
        # Calculate effective receptive field
        erf = (self.num_frames_per_filter - 1) * self.dilation_rate + 1
        input_length = inputs.shape[1]
        
        if input_length is not None and erf > input_length:
            raise ValueError(f"Effective receptive field ({erf}) exceeds input sequence length ({input_length}). "
                            f"Consider reducing kernel_size or dilation_rate.")
        
        # Add check for extreme values in inputs
        inputs = tf.where(tf.math.is_nan(inputs), tf.zeros_like(inputs), inputs)
        inputs = tf.clip_by_value(inputs, -1e6, 1e6)
        
        output = self.conv(inputs)
        
        # Check for NaN values in output and replace with zeros
        output = tf.where(tf.math.is_nan(output), tf.zeros_like(output), output)
        
        return output

@tf.keras.utils.register_keras_serializable()
class StatsLayer(layers.Layer):
    def call(self, inputs):
        # Use built-in TF functions with better numerical stability
        mean = tf.reduce_mean(inputs, axis=1)
        
        # Use a more numerically stable way to compute stddev
        # Add a small epsilon to variance before sqrt
        variance = tf.reduce_mean(tf.square(inputs - tf.expand_dims(mean, axis=1)), axis=1)
        std = tf.sqrt(tf.maximum(variance, 1e-12))
        
        # Check for NaN values
        mean = tf.where(tf.math.is_nan(mean), tf.zeros_like(mean), mean)
        std = tf.where(tf.math.is_nan(std), tf.ones_like(std), std)

        return tf.concat([mean, std], axis=1)

@tf.keras.utils.register_keras_serializable() 
class NormalizationLayer(layers.Layer):
    def call(self, inputs):
        return tf.linalg.normalize(inputs, axis=1)[0]  # Normalize along axis 1 and return the normalized tensor