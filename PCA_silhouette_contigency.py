#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 09:13:40 2024

@author: konstantinospapagoras
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pickle
import seaborn as sns
from sklearn.metrics import adjusted_mutual_info_score

# Load true labels and features from pickle file
with open('/Volumes/Seagate Expansion Drive/LSCC:LUAD:512NoBackroundFinal_Tiles/features_labels_InceptionV3.final.pkl', 'rb') as file:
    features, labels = pickle.load(file)

# Convert lists to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Verify the labels
unique_labels = np.unique(labels)
print(f"Unique labels: {unique_labels}")

# Perform PCA to reduce dimensionality
pca = PCA()
pca.fit(features)

# Determine number of components to retain 95% variance
desired_variance_retained = 0.95
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
num_components = np.argmax(cumulative_variance_ratio >= desired_variance_retained) + 1
print(f"Number of principal components to retain {desired_variance_retained * 100}% of variance: {num_components}")

# Perform PCA with the determined number of components
pca = PCA(n_components=num_components)
pca_components = pca.fit_transform(features)

# Apply K-means clustering and compute silhouette scores for k values 2 to 14
range_k = range(2, 15)
silhouette_scores = []

for k in range_k:
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(pca_components)
    silhouette_avg = silhouette_score(pca_components, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f"For n_clusters = {k}, the average silhouette score is {silhouette_avg}")

# Find the optimal k with the highest silhouette score
optimal_k = range_k[np.argmax(silhouette_scores)]
print(f"The optimal number of clusters is {optimal_k}")

# Plot silhouette scores
plt.figure(figsize=(10, 5))
plt.plot(range_k, silhouette_scores, marker='o')
plt.title('Silhouette Scores for K-means Clustering')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.xticks(range_k)

# Highlight the point with the best silhouette score with a red circle
best_k = optimal_k
best_score = silhouette_scores[np.argmax(silhouette_scores)]
plt.plot(best_k, best_score, marker='o', markersize=8, color='red')
plt.grid(False)
plt.show()

# Apply K-means clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
cluster_labels = kmeans.fit_predict(pca_components)

# Create a DataFrame with true labels and cluster labels
df = pd.DataFrame({'True Label': labels, 'Cluster Label': cluster_labels})

# Create a contingency table (cross-tabulation) to count the occurrences of each combination of true and cluster labels
contingency_table = pd.crosstab(df['True Label'], df['Cluster Label'], rownames=['True Label'], colnames=['Cluster Label'])

# Display the contingency table
print(contingency_table)

# Visualize the contingency table using a heatmap with a purple color map
plt.figure(figsize=(10, 8))
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Purples')
plt.title('Contingency Table of Cluster Labels vs True Labels')
plt.show()

# Define a consistent color palette for plotting
true_label_palette = sns.color_palette("Set1", len(unique_labels))
cluster_palette = sns.color_palette("husl", len(np.unique(cluster_labels)))

# Determine the subset size (60% of the data)
subset_size = int(0.6 * len(labels))
indices = np.random.choice(len(labels), size=subset_size, replace=False)

# Create a scatter plot for a subset of the true labels
plt.figure(figsize=(14, 6))

# Scatter plot for true labels
plt.subplot(1, 2, 1)
for i, label in enumerate(unique_labels):
    idx = labels[indices] == label
    plt.scatter(pca_components[indices][idx, 0], pca_components[indices][idx, 1], color=true_label_palette[i], label=label, alpha=0.7, s=30, marker='o')
plt.title('PCA with True Labels (Subset)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()

# Scatter plot for predicted clusters
plt.subplot(1, 2, 2)
for i, label in enumerate(np.unique(cluster_labels)):
    idx = cluster_labels == label
    plt.scatter(pca_components[idx, 0], pca_components[idx, 1], color=cluster_palette[i], label=f'Cluster {label}', alpha=0.7, marker='o', s=30)
plt.title('PCA with Cluster Predictions')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()

plt.show()

# Calculate the mutual information between true labels and cluster labels
mi_inception = adjusted_mutual_info_score(labels, cluster_labels)
print(f"Mutual Information for all features: {mi_inception}")

# Analyze mutual information for each cluster
for cluster in np.unique(cluster_labels):
    idx = cluster_labels == cluster
    mi_cluster = adjusted_mutual_info_score(labels[idx], cluster_labels[idx])
    print(f"Mutual Information for cluster {cluster}: {mi_cluster}")
