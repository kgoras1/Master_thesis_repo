#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 15:45:05 2024

@author: konstantinospapagoras
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from keras.preprocessing import image
import pickle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Load true labels and features from pickle file
with open('/Volumes/Seagate Expansion Drive/LSCC:LUAD nobgr kp algorithm/Features_Labels_Models/VGG16.pkl', 'rb') as file:
    features, labels = pickle.load(file)

# Convert lists to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Simplify the true labels
simplified_labels = np.array(['LSCC' if 'LSCC' in label else 'LUAD' for label in labels])
simplified_labels_u = np.unique(simplified_labels)

# Check the number of feature vectors and labels
print(f"Number of feature vectors: {len(features)}")
print(f"Number of labels: {len(labels)}")

# Define input directory containing the images
input_dir = '/Volumes/Seagate Expansion Drive/LSCC:LUAD nobgr kp algorithm/LSCC:LUAD:NoBackroundFinal_Tiles/'

# Function to load an image
def load_image(path):
    img = image.load_img(path, target_size=(224, 224))  # Assuming VGG16 input size
    return image.img_to_array(img)

# Function to recursively gather image paths from subdirectories
def get_image_paths(root_dir):
    image_paths = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('jpg', 'jpeg', 'png')):
                image_paths.append(os.path.join(root, file))
    return image_paths

# Get list of image paths
image_paths = get_image_paths(input_dir)
image_paths.sort()

# Load all images
images_list = [load_image(path) for path in image_paths]

# Check if the number of images matches the number of features
print(f"Number of images: {len(images_list)}")

# Debug information
if len(images_list) != len(features):
    print("Number of images does not match the number of features. Please verify the input directory and feature extraction process.")
else:
    # Perform PCA to reduce dimensionality
    desired_variance_retained = 0.95
    pca = PCA()
    pca.fit(features)
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    num_components = np.argmax(cumulative_variance_ratio >= desired_variance_retained) + 1
    print(f"Number of principal components to retain {desired_variance_retained * 100}% of variance: {num_components}")

    # Perform PCA with the determined number of components
    pca = PCA(n_components=num_components)
    pca_components = pca.fit_transform(features)

    # Plot PCA components
    plt.figure(figsize=(12, 10))
    sns.scatterplot(x=pca_components[:, 0], y=pca_components[:, 1], hue=simplified_labels, palette="deep", alpha=0.5)
    plt.title('PCA Plot with Labels as Colors')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Labels')

    # Select specific images to project onto the PCA space based on a criterion
    threshold = 30
    selected_indices = np.where(pca_components[:, 1] > threshold)[0]

    # Overlay images on PCA space
    for idx in selected_indices:
        img = images_list[idx]
        x, y = pca_components[idx, 0], pca_components[idx, 1]
        imagebox = OffsetImage(img, zoom=0.1)
        ab = AnnotationBbox(imagebox, (x, y), frameon=False)
        plt.gca().add_artist(ab)

    plt.savefig(os.path.join(input_dir, 'PCA_plot_with_images.png'))
    plt.show()




