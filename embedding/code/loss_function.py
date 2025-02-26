import tensorflow as tf
from tensorflow.keras.layers import Layer

class CustomLossLayer(Layer):
    def __init__(self, embedding_dim, **kwargs):
        super(CustomLossLayer, self).__init__(**kwargs)
        self.embedding_dim = embedding_dim
        # Initialize trainable weights S and b
        self.S = self.add_weight(
            name='S',
            shape=(embedding_dim, embedding_dim),
            initializer='random_normal',
            trainable=True
        )
        self.b = self.add_weight(
            name='b',
            shape=(1,),
            initializer='zeros',
            trainable=True
        )

    def call(self, embeddings, same_speaker):
        # Compute the similarity matrix
        similarity_matrix = tf.matmul(embeddings, embeddings, transpose_b=True)
        # Apply the transformation using S and b
        transformed_similarity = similarity_matrix - tf.matmul(tf.matmul(embeddings, self.S), embeddings, transpose_b=True)
        transformed_similarity += self.b
        # Apply sigmoid to the transformed similarity to get probabilities
        prob_same = tf.sigmoid(transformed_similarity)
        # Compute the loss for each pair in the batch
        loss_matrix = -same_speaker * tf.math.log(prob_same + 1e-10) - (1 - same_speaker) * tf.math.log(1 - prob_same + 1e-10)
        # Return the mean loss over all pairs
        self.add_loss(tf.reduce_mean(loss_matrix))
        return embeddings  # Return embeddings for further use