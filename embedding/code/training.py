import tensorflow as tf
import os, sys
import numpy as np


root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_path)

from  embedding.code.speaker_embedding_model import SpeakerEmbeddingModel

NUM_MFCC_COEFFICENTS = 48

dataset = np.load("data/audio_files/for_embedding_training/pairwise_numpy/pair_dataset.npz")
mfcc_1, mfcc_2, labels = dataset["mfcc_1"], dataset["mfcc_2"], dataset["labels"]

# Create a TensorFlow Dataset from the NumPy arrays
dataset = tf.data.Dataset.from_tensor_slices((mfcc_1, mfcc_2, labels))

batch_size = 48  # Adjust based on your system's capabilities

# Shuffle and batch the dataset
dataset = dataset.shuffle(buffer_size=len(labels))
dataset = dataset.batch(batch_size, drop_remainder=True)

# Initialize the model
model = SpeakerEmbeddingModel(input_dim=NUM_MFCC_COEFFICENTS, segment_length=10, embedding_dim=128)

# Train the model
model.train(dataset, epochs=40)
model.save_model('embedding/pretrained_models/embedding_model.h5')