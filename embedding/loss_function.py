import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as npw 

class CustomLoss(tf.keras.losses.Loss):
    def __init__(self, embedding_dim, name="custom_loss"):
        super().__init__(name=name)
        # Define S as a trainable matrix
        self.S = self.add_weight(
            name='S',
            shape=(embedding_dim, embedding_dim),
            initializer='random_normal',
            trainable=True
        )
        # Define b as a trainable vector
        self.b = self.add_weight(
            name='b',
            shape=(1,),  # Scalar bias for each pair
            initializer='zeros',
            trainable=True
        )

    def call(self, embeddings, same_speaker):
        """
        Compute the custom loss based on the probability that pairs of segments come from the same speaker.

        Args:
        - embeddings: Tensor of shape (batch_size, embedding_dim) representing the embeddings of the segments.
        - same_speaker: Tensor of shape (batch_size, batch_size) indicating whether each pair of segments is from the same speaker.

        Returns:
        - A tensor with the computed loss value.
        """
        # Compute the similarity matrix
        similarity_matrix = tf.matmul(embeddings, embeddings, transpose_b=True)

        # Apply the transformation using S and b
        transformed_similarity = similarity_matrix - tf.matmul(tf.matmul(embeddings, self.S), embeddings, transpose_b=True)
        transformed_similarity += self.b

        # Apply sigmoid to the transformed similarity to get probabilities
        prob_same = 1 / (1 + tf.exp(-transformed_similarity))

        # Compute the loss for each pair in the batch
        loss_matrix = -same_speaker * tf.math.log(prob_same) - (1 - same_speaker) * tf.math.log(1 - prob_same)

        # Return the mean loss over all pairs
        return tf.reduce_mean(loss_matrix)