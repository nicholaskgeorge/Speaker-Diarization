from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

DATA_PATH = r"C:\Users\nicok\Speaker-Diarization\data\training_Data\segmentation\numpy_files\clustering_data.npy"
DARA_LABELS = r"C:\Users\nicok\Speaker-Diarization\data\training_Data\segmentation\numpy_files\clustering_labels.npy"

SEGMENTATION_LENGTH = 0.25
DATA_POINT_LENGTH = 4
NUM_ITERATIONS = int(DATA_POINT_LENGTH/SEGMENTATION_LENGTH)

clustering_data = np.load(DATA_PATH)
clustering_labels = np.load(DARA_LABELS)
print(clustering_data)
point_start = 0
point_end = NUM_ITERATIONS

# Calculate WCSS for different number of clusters
min_clusters = 2
max_clusters = 4

for point in range(len(clustering_data)):
    # List to store silhouette scores
    silhouette_scores = []

    # Calculate silhouette score for different number of clusters
    for n_clusters in range(min_clusters, max_clusters+1):
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(clustering_data)
        cluster_labels = kmeans.labels_
        silhouette_avg = silhouette_score(clustering_data, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    # Find the optimal number of clusters based on silhouette score
    optimal_n_clusters = np.argmax(silhouette_scores) + min_clusters
    print("Optimal number of clusters based on silhouette score:", optimal_n_clusters)

    plt.plot(range(min_clusters, max_clusters+1), silhouette_scores, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Different Numbers of Clusters')
    plt.show()
    
    point_start += NUM_ITERATIONS
    point_end += NUM_ITERATIONS

