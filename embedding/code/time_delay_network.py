import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Define TDNN Layer
class TDNN(layers.Layer):
    def __init__(self, output_dim, context, input_dim, **kwargs):
        super(TDNN, self).__init__(**kwargs)
        self.context = context  # Time-splicing context (e.g., [-2, -1, 0, 1, 2])
        self.conv1d = layers.Conv1D(filters=output_dim, kernel_size=len(context),
                                    strides=1, padding='valid', activation='relu')
    
    def call(self, inputs):
        expanded_inputs = tf.concat([tf.roll(inputs, shift, axis=1) for shift in self.context], axis=-1)
        return self.conv1d(expanded_inputs)
    
class StatsLayer(layers.Layer):
    def call(self, inputs):
        mean = tf.reduce_mean(inputs, axis=1)
        std = tf.math.reduce_std(inputs, axis=1)
        return tf.concat([mean, std], axis=1)