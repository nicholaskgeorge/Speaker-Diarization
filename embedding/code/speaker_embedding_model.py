import tensorflow as tf
import numpy as np
from   tensorflow.keras import layers, Model
from   tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import HeNormal
tf.config.run_functions_eagerly(True)



# Import your custom layers
from time_delay_network import TDNN, StatsLayer, NormalizationLayer

class SpeakerEmbeddingModel:
    def __init__(self, input_dim=40, segment_length=10, embedding_dim=128):
        self.input_dim = input_dim
        self.segment_length = segment_length
        self.embedding_dim = embedding_dim
        self.model = self._build_embedding_model()
        self.optimizer = Adam(learning_rate=0.001)
        
        # Create trainable parameters for scoring function
        self.S = tf.Variable(
            initial_value=tf.random.normal(shape=(embedding_dim, embedding_dim), stddev=0.1),
            name='S', trainable=True
        )
        self.b = tf.Variable(
            initial_value=tf.zeros(shape=(1,)),
            name='b', trainable=True
        )

        self.frac_dataset_diff_speaker = 0.7

        # since there are many more examples of diff speakers than the same we introduce this 
        # constant to keep the effect of the same speaker cases the same
        self.diff_speaker_equalizer = self.frac_dataset_diff_speaker /(1-self.frac_dataset_diff_speaker )
        self.diff_speaker_equalizer = 1/self.diff_speaker_equalizer
    
    def _build_embedding_model(self):
        """Build the TDNN model that extracts embeddings from speech segments"""
        inputs = tf.keras.Input(shape=(self.segment_length, self.input_dim))
        
        # TDNN layers as described in the paper
        x = TDNN(output_dim=128 , dilation_rate=1, num_frames_per_filter=3, num_features=self.input_dim)(inputs)
        x = TDNN(output_dim=1024, dilation_rate=1, num_frames_per_filter=4, num_features=self.input_dim)(x)
        x = TDNN(output_dim=512, dilation_rate=2, num_frames_per_filter=3, num_features=self.input_dim)(x)
        
        # Stats pooling layer
        x = StatsLayer()(x)
        
        # Final dense layers
        x = layers.Dense(512, activation='relu', kernel_initializer=HeNormal())(x)
        embeddings = layers.Dense(self.embedding_dim, kernel_initializer=HeNormal())(x)

        # Apply custom normalization layer
        normalized_embeddings = NormalizationLayer()(embeddings)
        
        return Model(inputs=inputs, outputs=normalized_embeddings)
    
    def compute_similarity_score(self, embedding1, embedding2):
        """
        Compute the similarity score between two embeddings using the trainable S and b parameters
        as described in the paper: L(x,y) = x^T y - x^T Sx - y^T Sy + b
        """
        dot_product = tf.reduce_sum(embedding1 * embedding2, axis=1, keepdims=True)
        s_matrix_times_x = tf.matmul(self.S, tf.transpose(embedding1))
        x_transform = tf.reduce_sum(embedding1 * tf.transpose(s_matrix_times_x), axis=1, keepdims=True)
        s_matrix_times_y = tf.matmul(self.S, tf.transpose(embedding2))
        y_transform = tf.reduce_sum(embedding2 * tf.transpose(s_matrix_times_y), axis=1, keepdims=True)
        scores = dot_product - x_transform - y_transform + self.b
        return scores
    
    def compute_loss(self, embeddings1, embeddings2, same_speaker):
        """
        Compute the binary cross-entropy loss for the similarity scores
        same_speaker: 1 for same speaker pairs, 0 for different speaker pairs
        """
        scores = self.compute_similarity_score(embeddings1, embeddings2)
        probabilities = tf.sigmoid(scores)
        # Binary cross-entropy loss
        same_speaker = tf.cast(same_speaker, tf.float32)
        same_speaker = tf.expand_dims(same_speaker, axis=1) 
        epsilon = tf.keras.backend.epsilon()
        error_same = same_speaker * tf.math.log(probabilities+epsilon) 
        error_diff = (1 - same_speaker) * tf.math.log(1 - probabilities+epsilon) * self.diff_speaker_equalizer
        loss = -tf.reduce_mean(error_same + error_diff)
        return loss
    
    @tf.function
    def train_step(self, segment1_batch, segment2_batch, same_speaker_batch):
        """Single training step with gradient computation and parameter updates"""
        with tf.GradientTape() as tape:
            # Get embeddings for both segments
            embeddings1 = self.model(segment1_batch, training=True)
            embeddings2 = self.model(segment2_batch, training=True)
            # Compute loss
            loss = self.compute_loss(embeddings1, embeddings2, same_speaker_batch)
            print(f"loss is {loss}")
        
        # Get trainable variables from both the model and the similarity parameters
        trainable_vars = self.model.trainable_variables + [self.S, self.b]
        
        # Compute gradients
        gradients = tape.gradient(loss, trainable_vars)
        print(f"gradients are: {gradients[0][0][0]}")
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        return loss
    
    def train(self, train_dataset, epochs=10):
        """Train the model for a specified number of epochs"""
        
        for epoch in range(epochs):
            # Training
            train_loss = 0
            num_batches = 0
            
            for segment1_batch, segment2_batch, same_speaker_batch in train_dataset:
                print(f"batch number {num_batches}")
                batch_loss = self.train_step(segment1_batch, segment2_batch, same_speaker_batch)
                train_loss += batch_loss
                num_batches += 1
            
            train_loss /= num_batches
            
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")
    
    def load_best_model(self):
        """Load the best model weights and scoring parameters"""
        self.model.load_weights('best_speaker_embedding_model.h5')
        self.S.assign(np.load('best_S_matrix.npy'))
        self.b.assign(np.load('best_b_value.npy'))
    
    def get_embedding(self, segment):
        """Get embedding for a single speech segment"""
        # Add batch dimension if needed
        if len(segment.shape) == 2:
            segment = tf.expand_dims(segment, 0)
        
        return self.model(segment, training=False)