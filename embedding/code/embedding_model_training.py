import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import numpy as np
import sys
import os

# Get the absolute path of the "RPS" directory
path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(path)

from embedding.code.time_delay_network import TDNN, StatsLayer
from embedding.code.loss_function import CustomLossLayer


def build_tdnn_model(input_dim=40, segment_length=10, embedding_dim=128):
    inputs = keras.Input(shape=(segment_length, input_dim))
    x = TDNN(output_dim=512, context=[-2, -1, 0, 1, 2], input_dim=input_dim)(inputs)
    x = TDNN(output_dim=512, context=[-2, 0, 2], input_dim=512)(x)
    x = TDNN(output_dim=512, context=[-3, 0, 3], input_dim=512)(x)
    x = TDNN(output_dim=512, context=[0], input_dim=512)(x)
    x = TDNN(output_dim=1500, context=[0], input_dim=512)(x)
    x = StatsLayer()(x)  # Replace direct TensorFlow ops with the custom layer
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(512, activation='relu')(x)
    embeddings = layers.Dense(embedding_dim, activation=None)(x)
    same_speaker = keras.Input(shape=(None,), dtype=tf.float32)  # Placeholder for same_speaker labels
    outputs = CustomLossLayer(embedding_dim)(embeddings, same_speaker)
    model = keras.Model(inputs=[inputs, same_speaker], outputs=outputs, name="TDNN_SpeakerEmbedding")
    return model

# Build and compile the model
model = build_tdnn_model()
model.summary()

# Initialize the optimizer
optimizer = Adam(learning_rate=0.001)

@tf.function
def train_step(segment_1, segment_2, same_speaker):
    with tf.GradientTape() as tape:
        # Compute embeddings for both segments
        emb_1 = model([segment_1, same_speaker], training=True)  # Shape: (batch_size, 128)
        emb_2 = model([segment_2, same_speaker], training=True)  # Shape: (batch_size, 128)
        # Compute loss (already added in the CustomLossLayer)
        loss = sum(model.losses)
    # Compute gradients and update model weights
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Load dataset
dataset = np.load("data/audio_files/for_embedding_training/pairwise_numpy/pair_dataset.npz")
mfcc_1, mfcc_2, labels = dataset["mfcc_1"], dataset["mfcc_2"], dataset["labels"]
print("MFCC 1 Shape:", mfcc_1.shape)
print("MFCC 2 Shape:", mfcc_2.shape)
print("Labels Shape:", labels.shape)

# Training loop
epochs = 10
batch_size = 32

# Create TensorFlow dataset
dataset_tf = tf.data.Dataset.from_tensor_slices((mfcc_1, mfcc_2, labels)).batch(batch_size)

for epoch in range(epochs):
    epoch_loss = 0
    for pair_1, pair_2, same_speaker in dataset_tf:
        loss = train_step(pair_1, pair_2, same_speaker)
        epoch_loss += loss
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss.numpy()}")

# Save the entire model
model.save('embedding/pretrained_models/embeddings_tdnn.keras')