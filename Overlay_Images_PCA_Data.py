#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 15:45:05 2024

@author: konstantinospapagoras
"""

"""
PCA Image Visualization Script for LSCC vs LUAD Features.

This script loads VGG16-extracted features and corresponding labels 
for LSCC and LUAD histopathology images, applies PCA for dimensionality reduction, 
and visualizes the resulting components. It also overlays specific images on the PCA plot 
based on their distance from the origin.

Expected Input:
- Pickle file with extracted features and labels.
- Directory with original images corresponding to the features.

Output:
- PCA scatter plot with labels as colors.
- Optional images plotted in the reduced PCA space.

Author: [Your Name]
Date: [Date]
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from keras.preprocessing import image
import pickle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


# Load extracted features and labels
with open('/Volumes/Seagate Expansion Drive/LSCC:LUAD nobgr kp algorithm/Features_Labels_Models/VGG16.pkl', 'rb') as file:
    features, labels = pickle.load(file)

features = np.array(features)
labels = np.array(labels)

# Simplify labels to LSCC or LUAD
simplified_labels = np.array(['LSCC' if 'LSCC' in label else 'LUAD' for label in labels])
simplified_labels_u = np.unique(simplified_labels)

print(f"Number of feature vectors: {len(features)}")
print(f"Number of labels: {len(labels)}")

# Define image input directory
input_dir = '/Volumes/Seagate Expansion Drive/LSCC:LUAD nobgr kp algorithm/LSCC:LUAD:NoBackroundFinal_Tiles/'


def load_image(path):
    """
    Load and preprocess an image for visualization.
    
    Args:
        path (str): Path to the image.
        
    Returns:
        np.ndarray: Preprocessed image array.
    """
    img = image.load_img(path, target_size=(224, 224))
    return image.img_to_array(img)


def get_image_paths(root_dir):
    """
    Recursively retrieve image paths from a directory.
    
    Args:
        root_dir (str): Root directory containing images.
        
    Returns:
        list: List of image paths.
    """
    image_paths = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('jpg', 'jpeg', 'png')):
                image_paths.append(os.path.join(root, file))
    return image_paths


# Load image paths
image_paths = get_image_paths(input_dir)
image_paths.sort()

# Load all images
images_list = [load_image(path) for path in image_paths]

print(f"Number of images: {len(images_list)}")

# Check consistency between features and images
if len(images_list) != len(features):
    print("Warning: Number of images does not match the number of features.")
else:
    # PCA to retain 95% variance
    desired_variance_retained = 0.95
    pca = PCA()
    pca.fit(features)
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    num_components = np.argmax(cumulative_variance_ratio >= desired_variance_retained) + 1
    print(f"Number of components to retain {desired_variance_retained*100}% variance: {num_components}")

    pca = PCA(n_components=num_components)
    pca_components = pca.fit_transform(features)

    # Plot PCA Scatter
    plt.figure(figsize=(12, 10))
    sns.scatterplot(x=pca_components[:, 0], y=pca_components[:, 1],
                    hue=simplified_labels, palette="deep", alpha=0.5)
    plt.title('PCA Projection of Image Features')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Labels')

    # Optional: Select and plot images far from origin
    threshold = 30  # Radius threshold from origin
    distances = np.linalg.norm(pca_components[:, :2], axis=1)
    selected_indices = np.where(distances > threshold)[0]

    ax = plt.gca()
    for idx in selected_indices:
        img = images_list[idx] / 255.0  # Normalize
        imagebox = OffsetImage(img, zoom=0.3)
        ab = AnnotationBbox(imagebox, (pca_components[idx, 0], pca_components[idx, 1]), frameon=False)
        ax.add_artist(ab)

    plt.tight_layout()
    plt.show()
