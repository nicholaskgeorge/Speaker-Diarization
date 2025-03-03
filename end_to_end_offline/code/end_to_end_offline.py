import os
import numpy as np
import librosa
import tensorflow as tf
import sys
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_path)

from embedding.code.time_delay_network import TDNN, StatsLayer, NormalizationLayer

def extract_mfcc_matrices(directory):
    mfcc_matrices = []
    for filename in os.listdir(directory):
        if filename.endswith('.wav'):
            filepath = os.path.join(directory, filename)
            # Load the audio file
            y, sr = librosa.load(filepath, sr=None)
            # Ensure the audio file is exactly one second long
            if len(y) == sr:
                # Compute MFCCs with center=False
                mfcc = librosa.feature.mfcc(
                    y=y,
                    sr=sr,
                    n_mfcc=48,
                    n_fft=int(sr * 0.1),
                    hop_length=int(sr * 0.1),
                    center=False
                )
                # Transpose to get a 10x48 matrix
                mfcc_matrix = mfcc.T
                mfcc_matrices.append(mfcc_matrix)
            else:
                print(f"Skipping {filename}: Audio length is not exactly one second.")
    return mfcc_matrices

# Example usage
directory_path = 'end_to_end_offline/testing_aduio'
mfcc_matrices = extract_mfcc_matrices(directory_path)

custom_objects = {
    'TDNN': TDNN,
    'StatsLayer': StatsLayer,
    'NormalizationLayer': NormalizationLayer
}

model = load_model('embedding/pretrained_models/embedding_model.h5', custom_objects=custom_objects)

reshaped_data = np.expand_dims(mfcc_matrices[0], axis=0)

embeddings = []
for data in mfcc_matrices:
    reshaped_data = np.expand_dims(mfcc_matrices[0], axis=0)
    embeddings.append(model.predict(reshaped_data))

# Example: Replace with your actual 128-dimensional embeddings
embeddings = np.array(embeddings)

embeddings = embeddings.reshape(embeddings.shape[0], -1)

print(embeddings.shape)

# Step 1: Compute pairwise cosine distances
distance_matrix = cosine_distances(embeddings)

# Step 2: Perform Agglomerative Clustering
clustering = AgglomerativeClustering(
    n_clusters=None,  # Let the algorithm decide the number of clusters
    distance_threshold=5,  # Adjust based on your data
    linkage='average'
)
clustering.fit(distance_matrix)

# Step 3: Visualize the Dendrogram to determine the optimal threshold
linked = linkage(distance_matrix, 'average')
plt.figure(figsize=(10, 7))
dendrogram(linked)
plt.show()

# The clustering.labels_ will contain the cluster assignments
print("Cluster assignments:", clustering.labels_)
print("Number of clusters:", len(set(clustering.labels_)))

