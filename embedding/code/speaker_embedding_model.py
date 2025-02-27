import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K



import numpy as np
import os
import sys
import glob

# Import your custom layers
from time_delay_network import TDNN, StatsLayer

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
    
    def _build_embedding_model(self):
        """Build the TDNN model that extracts embeddings from speech segments"""
        inputs = tf.keras.Input(shape=(self.segment_length, self.input_dim))
        
        # TDNN layers as described in the paper
        x = TDNN(output_dim=512, context=[-1, 0, 1], input_dim=self.input_dim)(inputs)
        x = TDNN(output_dim=512, context=[-2, 0, 2], input_dim=512)(x)
        x = TDNN(output_dim=512, context=[-3, 0, 3], input_dim=512)(x)
        x = TDNN(output_dim=512, context=[-3, 0, 3], input_dim=512)(x)
        
        # Stats pooling layer
        x = StatsLayer()(x)
        
        # Final dense layers
        x = layers.Dense(512, activation='relu')(x)
        embeddings = layers.Dense(self.embedding_dim)(x)
        
        # # Normalize the embeddings
        # normalized_embeddings = K.l2_normalize(embeddings, axis=1)
        
        return Model(inputs=inputs, outputs=embeddings)
    
    def compute_similarity_score(self, embedding1, embedding2):
        """
        Compute the similarity score between two embeddings using the trainable S and b parameters
        as described in the paper: L(x,y) = x^T y - x^T Sx - y^T Sy + b
        """
        dot_product = tf.reduce_sum(embedding1 * embedding2, axis=1, keepdims=True)
        x_transform = tf.reduce_sum(embedding1 * tf.matmul(embedding1, self.S), axis=1, keepdims=True)
        y_transform = tf.reduce_sum(embedding2 * tf.matmul(embedding2, self.S), axis=1, keepdims=True)
        
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
        loss = -tf.reduce_mean(
            same_speaker * tf.math.log(probabilities + 1e-10) + 
            (1 - same_speaker) * tf.math.log(1 - probabilities + 1e-10)
        )
        
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
        
        # Get trainable variables from both the model and the similarity parameters
        trainable_vars = self.model.trainable_variables + [self.S, self.b]
        
        # Compute gradients
        gradients = tape.gradient(loss, trainable_vars)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        return loss
    
    def train(self, train_dataset, epochs=10):
        """Train the model for a specified number of epochs"""
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            train_loss = 0
            num_batches = 0
            
            for segment1_batch, segment2_batch, same_speaker_batch in train_dataset:
                batch_loss = self.train_step(segment1_batch, segment2_batch, same_speaker_batch)
                train_loss += batch_loss
                num_batches += 1
            
            train_loss /= num_batches
            
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")
            
            # # Save the best model
            # if val_loss < best_val_loss:
            #     best_val_loss = val_loss
            #     self.model.save_weights('best_speaker_embedding_model.h5')
            #     # Save S and b parameters
            #     np.save('best_S_matrix.npy', self.S.numpy())
            #     np.save('best_b_value.npy', self.b.numpy())
            #     print(f"Model saved at epoch {epoch+1}")
    
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

# # Load dataset
# dataset = np.load("data/audio_files/for_embedding_training/pairwise_numpy/pair_dataset.npz")
# mfcc_1, mfcc_2, labels = dataset["mfcc_1"], dataset["mfcc_2"], dataset["labels"]


# model.save('embedding/pretrained_models/embeddings_tdnn.keras')