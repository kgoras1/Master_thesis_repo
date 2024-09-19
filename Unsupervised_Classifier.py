#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 10:23:29 2024

@author: konstantinospapagoras
"""
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib.lines import Line2D

# Load true labels and features from pickle file
with open('/Volumes/Seagate Expansion Drive/HistoEncoder/Histoencoder_features_labels.final.pkl', 'rb') as file:
    features, labels = pickle.load(file)

# Convert lists to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Define training and testing slides
train_slides = [
    'LSCC_1', 'LSCC_2', 'LSCC_3', 'LSCC_5', 'LSCC_6', 'LSCC_7',  
    'LUAD_1', 'LUAD_2', 'LUAD_3', 'LUAD_4', 'LUAD_5', 'LUAD_7'
]
test_slides = ['LSCC_4', 'LUAD_6']

# Assign different colors for each test WSI
test_slide_colors = {
    'LSCC_4': '#ffff00',  # Yellow for LSCC_4
    'LUAD_6': '#000000'   # Black for LUAD_6
}

# Extract training features and labels
train_features, train_labels = [], []
for slide in train_slides:
    slide_mask = labels == slide  # Create mask for current slide
    train_features.append(features[slide_mask])
    train_labels.append(labels[slide_mask])

# Concatenate training data
train_features = np.concatenate(train_features, axis=0)
train_labels = np.concatenate(train_labels, axis=0)

# Extract testing features and labels
test_features, test_labels = [], []
for slide in test_slides:
    slide_mask = labels == slide  # Create mask for current slide
    test_features.append(features[slide_mask])
    test_labels.append(labels[slide_mask])

# Concatenate testing data
test_features = np.concatenate(test_features, axis=0)
test_labels = np.concatenate(test_labels, axis=0)

# Perform PCA on training features
pca = PCA(n_components=0.95)  # Retain 95% of variance
train_features_pca = pca.fit_transform(train_features)

# Fit KMeans clustering on PCA-reduced training features
n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(train_features_pca)

# Transform test features using the same PCA
test_features_pca = pca.transform(test_features)

# Predict clusters for the test features
test_tile_labels = kmeans.predict(test_features_pca)

# Count distribution of tiles in test slides across clusters
tile_distribution = Counter(test_tile_labels)
print(f"Test tile distribution across clusters: {tile_distribution}")

# Combine training and test data for full analysis
full_features = np.concatenate([train_features, test_features], axis=0)
full_labels = np.concatenate([train_labels, test_labels], axis=0)
full_features_pca = pca.transform(full_features)

# Predict clusters for combined features
full_tile_labels = kmeans.predict(full_features_pca)

# Create a label map to group slides into LSCC and LUAD
group_labels = np.array(['LSCC' if 'LSCC' in label else 'LUAD' for label in full_labels])

# Define colors for clusters and true labels
cluster_colors = ['#0000ff', '#ff0000']  # Blue and red for clusters
label_colors = {'LSCC': '#ff0000', 'LUAD': '#0000ff'}  # Red and blue for true labels

# Create the figure and subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Subplot 1: Original data grouped into LSCC and LUAD
for group, color in label_colors.items():
    group_mask = group_labels == group
    ax1.scatter(full_features_pca[group_mask, 0], full_features_pca[group_mask, 1], 
                color=color, label=group, s=10, alpha=0.7)

ax1.set_title('PCA Visualization Grouped by LSCC and LUAD')
ax1.set_xlabel('PCA Component 1')
ax1.set_ylabel('PCA Component 2')
ax1.legend()

# Subplot 2: Test points highlighted with their predicted cluster
train_mask = np.isin(full_labels, train_slides)  # Mask for training data
test_mask = np.isin(full_labels, test_slides)    # Mask for test data

# Plot training points with their cluster labels
ax2.scatter(full_features_pca[train_mask, 0], full_features_pca[train_mask, 1], 
            c=[cluster_colors[label] for label in kmeans.labels_], s=10, label='Training Data')

# Highlight test points with new colors based on their WSI (but keep cluster-based color for consistency)
for test_slide in test_slides:
    current_test_mask = full_labels == test_slide  # Apply mask to concatenated `full_labels`
    test_color = test_slide_colors[test_slide]
    ax2.scatter(full_features_pca[current_test_mask, 0], full_features_pca[current_test_mask, 1], 
                color=test_color, s=50, label=f'Test Data ({test_slide})', edgecolors='black')

# Annotate the test points with cluster labels
for i, (x, y) in enumerate(zip(full_features_pca[test_mask, 0], full_features_pca[test_mask, 1])):
    ax2.annotate(test_tile_labels[i], (x, y), fontsize=9, color='black', ha='center', va='center')

# Add legend for clusters and test slides
legend_lines = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=f'Cluster {i}') for i, color in enumerate(cluster_colors)
]
test_legend_lines = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=f'{slide}') for slide, color in test_slide_colors.items()
]

ax2.legend(handles=legend_lines + test_legend_lines, loc='best')

ax2.set_title('PCA Visualization with Highlighted Test Data')
ax2.set_xlabel('PCA Component 1')
ax2.set_ylabel('PCA Component 2')

plt.show()
