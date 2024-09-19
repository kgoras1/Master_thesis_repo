#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 15:59:44 2024

@author: konstantinospapagoras
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
import pickle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import numpy as np
import pandas as pd
import pickle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Initialize VGG16 model for feature extraction
model = VGG16(weights='imagenet', include_top=True)
feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)
print("\nFeature Extractor Model Summary:")
feat_extractor.summary()

# Define input directory containing the subdirectories
input_dir = '/Volumes/Seagate Expansion Drive/data/'

# Get list of subdirectories
subdirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

# Initialize lists to store data
features = []
labels = []
images_list = []
original_means = []
original_stds = []

def load_image(path):
    img = image.load_img(path, target_size=model.input_shape[1:3])
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x

# Process each subdirectory
for subdir in subdirs:
    subdir_path = os.path.join(input_dir, subdir)
    images = [os.path.join(subdir_path, f) for f in os.listdir(subdir_path) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
    print(f"Processing {len(images)} images in {subdir}...")

    # Load images and calculate statistics
    imgs = [load_image(img_path)[0] for img_path in images]

    images_list.extend(imgs)  # Collecting images for later use



# Load true labels and predicted labels from pickle file
features, labels = pickle.load(open('/Volumes/Seagate Expansion Drive/data/features_labels.pkl', 'rb'))

# Convert lists to numpy arrays
features = np.array(features)
labels = np.array(labels)


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
pca_features = pca.fit_transform(features)

print("Original features shape:", features.shape)
print("Transformed features shape:", pca_features.shape)

# Plot PCA components
unique_labels = np.unique(labels)
custom_palette = {label: sns.color_palette()[i] for i, label in enumerate(unique_labels)}

plt.figure(figsize=(12, 10))
sns.scatterplot(x=pca_features[:, 0], y=pca_features[:, 1], hue=labels, palette=custom_palette, alpha=0.5)
plt.title('PCA Plot with Labels as Colors')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Labels')

# Overlay images on PCA space
for label in unique_labels:
    class_indices = np.where(labels == label)[0]
    selected_indices = np.random.choice(class_indices, size=30, replace=False)
    for idx in selected_indices:
        img = images_list[idx]
        x, y = pca_features[idx, 0], pca_features[idx, 1]
        # Use offset_image to avoid plotting image directly using imshow
        imagebox = OffsetImage(img, zoom=0.1)
        ab = AnnotationBbox(imagebox, (x, y), frameon=False)
        plt.gca().add_artist(ab)

plt.savefig(os.path.join(input_dir, 'PCA_plot_with_images_version3.png'))
plt.show()