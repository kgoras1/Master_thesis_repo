#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 11:16:45 2024

@author: konstantinospapagoras
"""

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import cv2
import os

def load_images(directory):
    img_path = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpeg') or filename.endswith('.jpg'):
            img_path.append(os.path.join(directory, filename))
            
    images = []
    for path in img_path:
        img = cv2.imread(path)
        img = cv2.resize(img, (512, 512))
      
        images.append(img)
    images = np.array(images)
    images = images.astype('float32') / 255.0
    
    # Reshape images to have one row per image
    num_images, height, width, channels = images.shape
    flatten_images = images.reshape(num_images, -1)  # Flatten each image
    
    print("Shape of flatten_images:", flatten_images.shape)  # Add this line to check the shape
    
    # Ensure the number of features matches expected dimensions
    num_features = height * width * channels
    assert flatten_images.shape[1] == num_features, f"Expected {num_features} features, but got {flatten_images.shape[1]}"
    
    # Calculate mean image
    mean_image = np.mean(flatten_images, axis=0)
    
    # Center the images
    centered_images = flatten_images - mean_image
    
    print("Shape of centered_images:", centered_images.shape)  # Add this line to check the shape
    
    return centered_images, mean_image, img_path




def plot_images(original_images, reconstructed_images, num_images=97):
    fig, axes = plt.subplots(num_images, 2, figsize=(10, 10))
    for i in range(num_images):
        original_img = np.clip(original_images[i], 0, 1)  # Clip image data
        reconstructed_img = np.clip(reconstructed_images[i], 0, 1)  # Clip image data
        axes[i, 0].imshow(original_img.reshape(512, 512, 3))
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')
        axes[i, 1].imshow(reconstructed_img.reshape(512, 512, 3))
        axes[i, 1].set_title('Reconstructed')
        axes[i, 1].axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    directory = input("Enter the path to the directory containing images: ")
    if not os.path.isdir(directory):
        print("Invalid directory path")
        exit()
    centered_images, mean_image, img_paths = load_images(directory)
    
    # Perform PCA
    pca = PCA()
    pca.fit(centered_images)
    
    # Determine number of components to retain 95% variance
    desired_variance_retained = 0.95 
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    num_components = np.argmax(cumulative_variance_ratio >= desired_variance_retained)
    print("Number of principal components to retain {} of variance: {}".format(desired_variance_retained * 100, num_components))
    
    # Reconstruct images using retained components
    reconstructed_images = np.dot(pca.transform(centered_images[:, :786432]), pca.components_[:786432, :]) + mean_image
    
    # Plot original and reconstructed images
    plot_images(centered_images[:97], reconstructed_images[:97])
