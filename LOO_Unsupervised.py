#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 12:07:44 2024

@author: konstantinospapagoras
"""

import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import Counter

# Load true labels and features from pickle file
with open('/Volumes/Seagate Expansion Drive/LSCC:LUAD nobgr kp algorithm/Features_Labels_Models/VGG16._final_correctpkl', 'rb') as file:
    features, labels = pickle.load(file)

# Convert lists to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Define all WSI slides
all_slides = [
    'LSCC_1', 'LSCC_2', 'LSCC_3', 'LSCC_4', 'LSCC_5', 'LSCC_6', 'LSCC_7',
    'LUAD_1', 'LUAD_2', 'LUAD_3', 'LUAD_4', 'LUAD_5', 'LUAD_6', 'LUAD_7'
]

# Initialize KMeans parameters
n_clusters = 2
    
# Iterate over each slide, leaving one out
for test_slide in all_slides:
    print(f"\nProcessing Leave-One-Out for WSI: {test_slide}")
    
    # Separate training and test data based on the leave-one-out method
    train_slides = [slide for slide in all_slides if slide != test_slide]
    
    # Extract training features and labels
    train_features, train_labels = [], []
    for slide in train_slides:
        slide_mask = labels == slide  # Create mask for current slide
        train_features.append(features[slide_mask])
        train_labels.append(labels[slide_mask])
    
    # Concatenate training data
    train_features = np.concatenate(train_features, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    
    # Extract test features and labels
    test_mask = labels == test_slide  # Create mask for the test slide
    test_features = features[test_mask]
    test_labels = labels[test_mask]
    
    # Perform PCA on training features
    pca = PCA(n_components=0.95)  # Retain 95% of variance
    train_features_pca = pca.fit_transform(train_features)
    
    # Fit KMeans clustering on PCA-reduced training features
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(train_features_pca)
    
    # Transform test features using the same PCA
    test_features_pca = pca.transform(test_features)
    
    # Predict clusters for the test features
    test_tile_labels = kmeans.predict(test_features_pca)
    
    # Count and print the number of tiles assigned to each cluster
    tile_counts = Counter(test_tile_labels)
    print(f"Number of tiles assigned to each cluster for {test_slide}: {tile_counts}")