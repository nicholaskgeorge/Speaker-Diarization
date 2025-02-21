import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class EmbeddingScoring(layers.Layer):
    def __init__(self, embedding_dim):
        super(EmbeddingScoring, self).__init__()
        self.embedding_dim = embedding_dim
        
    def build(self, input_shape):
        # Initialize S matrix - will be made symmetric
        self.S = self.add_weight(
            name='S',
            shape=(self.embedding_dim, self.embedding_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        
        # Initialize scalar offset b
        self.b = self.add_weight(
            name='b',
            shape=(),  # scalar
            initializer='zeros',
            trainable=True
        )
    
    def make_symmetric(self):
        # Make S symmetric by averaging with its transpose
        return (self.S + tf.transpose(self.S)) / 2.0
    
    def score_pairs(self, x, y):
        # Get symmetric S matrix
        S = self.make_symmetric()
        
        # Calculate L(x,y) = x^T y - x^T S x - y^T S y + b
        # From equation 2 in the paper
        xy = tf.reduce_sum(x * y, axis=-1)  # x^T y
        xSx = tf.reduce_sum(x * tf.matmul(x, S), axis=-1)  # x^T S x
        ySy = tf.reduce_sum(y * tf.matmul(y, S), axis=-1)  # y^T S y
        
        return xy - xSx - ySy + self.b