from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import os
import sys
from sklearn.cluster import KMeans

sys.path.append(os.path.abspath(r"file_movement_scripts"))
from get_files_in_folder import get_files_in_folder

DATA_PATH = r"data\training_data\segmentation\numpy_files\cluster_data"
LABEL_PATH = r"data\training_data\segmentation\numpy_files\cluster_labels\all_labels.npy"

files = get_files_in_folder(DATA_PATH)
clustering_labels = np.load(LABEL_PATH)


def kmeans_and_plot(X, max_clusters=4):
    wss = []
    for k in range(1, max_clusters):
        kmeans = KMeans(init="random", n_clusters=k, n_init=10, random_state=1)
        kmeans.fit(X)
        wss.append(kmeans.inertia_)  # Inertia is the WSS
    plt.plot(range(1, max_clusters), wss, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares (WSS)')
    plt.title('Elbow Method for Optimal k')
    plt.show()


for index, data_point in enumerate(files):
    # List to store silhouette scores
    silhouette_scores = []
    matrix_path = os.path.join(DATA_PATH, data_point)
    clustering_data = np.load(matrix_path)
    print(f"Current label is {clustering_labels[index]}")
    kmeans_and_plot(clustering_data)
    

    

