#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 17:29:14 2024

@author: konstantinospapagoras
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pickle
import seaborn as sns

# Load true labels and features from pickle file
with open('/Volumes/Seagate Expansion Drive/LSCC:LUAD nobgr kp algorithm/Features_Labels_Models/VGG16.pkl', 'rb') as file:
    features, labels = pickle.load(file)

# Convert lists to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Simplify the true labels
simplified_labels = np.array(['LSCC' if 'LSCC' in label else 'LUAD' for label in labels])
simplified_labels_u = np.unique(simplified_labels)

# Print unique labels
unique_labels = np.unique(labels)
print(f"Unique labels: {unique_labels}")
print(f"Simplified labels: {simplified_labels_u}")

# Shuffle the features across columns
np.random.seed(42)  # For reproducibility
shuffled_features = np.copy(features)  # Make a copy to shuffle
np.random.shuffle(shuffled_features.T)  # Shuffle the columns (features)

# Perform PCA and retain components that explain 95% of the variance
pca = PCA()
pca.fit(shuffled_features)
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
num_components = np.argmax(cumulative_variance_ratio >= 0.95) + 1
print(f"Number of components to retain 95% variance: {num_components}")

# Apply PCA to reduce to the determined number of components
pca = PCA(n_components=num_components)
pca_components = pca.fit_transform(shuffled_features)

# Apply K-means clustering
kmeans = KMeans(n_clusters=2, random_state=42)
cluster_labels = kmeans.fit_predict(pca_components)

# Plot the first two PCA components with the clustering results
plt.figure(figsize=(14, 6))

# Scatter plot for true labels
plt.subplot(1, 2, 1)
for label in simplified_labels_u:
    idx = simplified_labels == label
    plt.scatter(pca_components[idx, 0], pca_components[idx, 1], label=label, alpha=0.2, s=10)
plt.title('PCA with Simplified True Labels (Shuffled Features)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()

# Scatter plot for cluster labels
plt.subplot(1, 2, 2)
for label in np.unique(cluster_labels):
    idx = cluster_labels == label
    plt.scatter(pca_components[idx, 0], pca_components[idx, 1], label=f'Cluster {label}', alpha=0.2, s=10)
plt.title('PCA with Cluster Predictions (Shuffled Features)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()

plt.show()

