import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load dataset (use your dataset here)
data = pd.read_csv('customer_data.csv')

# Preprocessing: Select features and scale them
X = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)  # Adjust parameters as necessary
dbscan_labels = dbscan.fit_predict(X_scaled)

# Evaluate clustering quality using silhouette score
kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
dbscan_silhouette = silhouette_score(X_scaled, dbscan_labels) if len(set(dbscan_labels)) > 1 else -1

# Visualize the clusters for both methods (2D plot)
plt.figure(figsize=(12, 6))

# K-Means clusters
plt.subplot(1, 2, 1)
plt.scatter(X_scaled[kmeans_labels == 0, 0], X_scaled[kmeans_labels == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X_scaled[kmeans_labels == 1, 0], X_scaled[kmeans_labels == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X_scaled[kmeans_labels == 2, 0], X_scaled[kmeans_labels == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(X_scaled[kmeans_labels == 3, 0], X_scaled[kmeans_labels == 3, 1], s=100, c='purple', label='Cluster 4')
plt.scatter(X_scaled[kmeans_labels == 4, 0], X_scaled[kmeans_labels == 4, 1], s=100, c='orange', label='Cluster 5')
plt.title("K-Means Clustering")
plt.xlabel('Age (scaled)')
plt.ylabel('Annual Income (scaled)')
plt.legend()

# DBSCAN clusters
plt.subplot(1, 2, 2)
plt.scatter(X_scaled[dbscan_labels == 0, 0], X_scaled[dbscan_labels == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X_scaled[dbscan_labels == 1, 0], X_scaled[dbscan_labels == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X_scaled[dbscan_labels == -1, 0], X_scaled[dbscan_labels == -1, 1], s=100, c='gray', label='Noise')
plt.title("DBSCAN Clustering")
plt.xlabel('Age (scaled)')
plt.ylabel('Annual Income (scaled)')
plt.legend()

# Show plots
plt.tight_layout()
plt.show()

# Print silhouette scores
print(f"Silhouette Score for K-Means: {kmeans_silhouette}")
print(f"Silhouette Score for DBSCAN: {dbscan_silhouette}")