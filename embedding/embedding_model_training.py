import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import numpy as np
from embedding.time_delay_network import TDNN
from embedding.loss_function import CustomLoss



# Define the TDNN-based Speaker Embedding Model
def build_tdnn_model(input_dim=40, sequence_length=100):
    inputs = keras.Input(shape=(sequence_length, input_dim))
    
    x = TDNN(output_dim=512, context=[-2, -1, 0, 1, 2], input_dim=input_dim)(inputs)
    x = TDNN(output_dim=512, context=[-2, 0, 2], input_dim=512)(x)
    x = TDNN(output_dim=512, context=[-3, 0, 3], input_dim=512)(x)
    x = TDNN(output_dim=512, context=[0], input_dim=512)(x)
    x = TDNN(output_dim=1500, context=[0], input_dim=512)(x)
    
    mean = tf.reduce_mean(x, axis=1)
    std = tf.math.reduce_std(x, axis=1)
    x = tf.concat([mean, std], axis=1)
    
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(512, activation='relu')(x)
    outputs = layers.Dense(128, activation=None)(x)  # 128-dimensional speaker embedding
    
    model = keras.Model(inputs, outputs, name="TDNN_SpeakerEmbedding")
    return model


# Build and compile the model
model = build_tdnn_model()
model.summary()


# Initialize the model and optimizer
model = build_tdnn_model(input_dim=40, sequence_length=100)
optimizer = Adam(learning_rate=0.001)

# Define the training step
@tf.function
def train_step(pair_1, pair_2, same_speaker):
    with tf.GradientTape() as tape:
        # Forward pass: compute embeddings for both pairs
        emb_1 = model(pair_1, training=True)
        emb_2 = model(pair_2, training=True)
        
        # Compute the loss
        loss = CustomLoss(emb_1, emb_2, same_speaker)
    
    # Compute gradients and update model weights
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Training loop
def train(dataset, epochs):
    for epoch in range(epochs):
        epoch_loss = 0
        for pair_1, pair_2, same_speaker in dataset:
            loss = train_step(pair_1, pair_2, same_speaker)
            epoch_loss += loss
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss.numpy()}")