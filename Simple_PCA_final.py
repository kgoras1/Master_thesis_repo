#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 09:56:53 2024

@author: konstantinospapagoras
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pickle
import seaborn as sns

# Define the labels of interest
labels_of_interest = {'LSCC_1', 'LSCC_2', 'LUAD_2', 'LUAD_3'}


# Load true labels and features from pickle file
with open('/Volumes/Seagate Expansion Drive/LSCC:LUAD nobgr kp algorithm/Features_Labels_Models/VGG16._final_correctpkl', 'rb') as file:  # Update the path to your pickle file
    features, labels = pickle.load(file)

# Convert lists to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Filter data to keep only the labels of interest
mask = np.isin(labels, list(labels_of_interest))
filtered_features = features[mask]
filtered_labels = labels[mask]

# Verify the labels
unique_labels = np.unique(filtered_labels)
print(f"Filtered labels: {unique_labels}")



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

# Print variance explained by the first two principal components
print(f"Variance explained by the first principal component: {pca.explained_variance_ratio_[0]:.2f}")
print(f"Variance explained by the second principal component: {pca.explained_variance_ratio_[1]:.2f}")

# Plotting
plt.figure(figsize=(12, 6))

# Subplot 1: All points in PCA space (grey color)
plt.subplot(1, 2, 1)
plt.scatter(pca_components[:, 0], pca_components[:, 1], alpha=0.5, s=30, c='grey')
plt.title("PCA Projection (All Points)")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.grid(False)

# Subplot 2: Points in PCA space with simplified labels (LUAD and LSCC)
plt.subplot(1, 2, 2)
sns.scatterplot(x=pca_components[:, 0], y=pca_components[:, 1], hue=filtered_labels, alpha=0.7, s=30, palette={'LUAD': 'blue', 'LSCC': 'red'})
plt.title("PCA Projection (Simplified True Labels)")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.legend(title="Simplified Labels")
plt.grid(False)

# Show the plot
plt.tight_layout()
plt.show()
