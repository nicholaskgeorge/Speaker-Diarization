import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from embedded_scoring import EmbeddingScoring

class NetworkInNetwork(layers.Layer):
    def __init__(self, num_networks=50, hidden_dim=1000, output_dim=500):
        super(NetworkInNetwork, self).__init__()
        self.num_networks = num_networks
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
    def build(self, input_shape):
        # Create parameters for n micro neural networks with tied parameters
        self.layer1 = layers.Dense(self.hidden_dim)
        self.layer2 = layers.Dense(self.hidden_dim)
        self.layer3 = layers.Dense(self.output_dim)
        
    def call(self, x):
        outputs = []
        for _ in range(self.num_networks):
            # Each micro network is 3 layers of ReLU + affine
            h = tf.nn.relu(self.layer1(x))
            h = tf.nn.relu(self.layer2(h))
            h = tf.nn.relu(self.layer3(h))
            outputs.append(h)
            
        # Combine outputs from all micro networks
        return tf.reduce_mean(outputs, axis=0)

class TemporalPooling(layers.Layer):
    def __init__(self):
        super(TemporalPooling, self).__init__()
        
    def call(self, x):
        # Average pooling over time dimension
        return tf.reduce_mean(x, axis=1)

class SpeakerDiarizationModel(Model):
    def __init__(self, embedding_dim=400):
        super(SpeakerDiarizationModel, self).__init__()
        
        # [Previous layers remain the same...]
        self.td1 = layers.Conv1D(150, kernel_size=3, padding='same')
        self.td2 = layers.Conv1D(150, kernel_size=4, padding='same')
        self.td3 = layers.Conv1D(150, kernel_size=7, padding='same')
        self.td4 = layers.Conv1D(150, kernel_size=7, padding='same')
        
        self.nin1 = NetworkInNetwork()
        self.nin2 = NetworkInNetwork()
        self.nin3 = NetworkInNetwork()
        self.nin4 = NetworkInNetwork()
        
        self.temporal_pooling = TemporalPooling()
        self.hidden5 = layers.Dense(500, activation='relu')
        self.embedding = layers.Dense(embedding_dim)
        
        # Scoring mechanism
        self.scoring = EmbeddingScoring(embedding_dim)
        
    def call(self, inputs, training=False):
        # Get embeddings through the network
        x = self.td1(inputs)
        x = tf.nn.relu(x)
        x = self.nin1(x)
        # [... other layers remain the same ...]
        embeddings = self.embedding(x)
        
        if training:
            # During training, return both embeddings and scores
            return embeddings
        else:
            # During inference, just return embeddings
            return embeddings