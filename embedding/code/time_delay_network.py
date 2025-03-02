import tensorflow as tf
from tensorflow.keras import layers

class TDNN(layers.Layer):
    def __init__(self, output_dim, dilation_rate, num_frames_per_filter, num_features, **kwargs):
        super(TDNN, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.dilation_rate = dilation_rate
        self.num_frames_per_filter =num_frames_per_filter
        self.num_features = num_features
        self.conv = layers.Conv1D(
                                  filters=self.output_dim,
                                  kernel_size=self.num_frames_per_filter,
                                  dilation_rate=self.dilation_rate,
                                  strides=1, 
                                  padding='valid', 
                                  activation='relu'
                                 )

    def call(self, inputs):
         # Calculate effective receptive field
        erf = (self.num_frames_per_filter - 1) * self.dilation_rate + 1
        input_length = inputs.shape[1]

        if input_length is not None and erf > input_length:
            raise ValueError(f"Effective receptive field ({erf}) exceeds input sequence length ({input_length}). "
                             f"Consider reducing kernel_size or dilation_rate.")
        
        return self.conv(inputs)

class StatsLayer(layers.Layer):
    def call(self, inputs):

        mean = tf.reduce_mean(inputs, axis=1)
        std = tf.math.reduce_std(inputs, axis=1)
        return tf.concat([mean, std], axis=1)