#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 09:29:50 2024

@author: konstantinospapagoras
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import Counter
import pickle
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Define input directory containing the subdirectories
input_dir = '/Volumes/Seagate Expansion Drive/LSCC:LUAD:512NoBackroundFinal_Tiles/'

# Get list of subdirectories
subdirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

# Initialize lists to store data
features = []
labels = []
images_list = []

def load_image(path):
    img = image.load_img(path, target_size=(224, 224))  # Assuming model expects 224x224 input
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x

# Process each subdirectory
for subdir in subdirs:
    subdir_path = os.path.join(input_dir, subdir)
    images = [os.path.join(subdir_path, f) for f in os.listdir(subdir_path) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
    print(f"Processing {len(images)} images in {subdir}...")

    # Load images
    imgs = [load_image(img_path)[0] for img_path in images]
    images_list.extend(imgs)  # Collecting images for later use

# Print the number of images loaded
print(f"Total images loaded: {len(images_list)}")

# Load true labels and features from pickle file
with open('/Volumes/Seagate Expansion Drive/LSCC:LUAD nobgr kp algorithm/Features_Labels_Models/VGG16.pkl', 'rb') as file:
    features, labels = pickle.load(file)

# Convert lists to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Ensure the number of images matches the number of features
if len(images_list) != len(features):
    print(f"Warning: Number of images ({len(images_list)}) does not match number of features ({len(features)}).")

# Map to simplified classes: LSCC and LUAD
simplified_labels = np.array(['LSCC' if 'LSCC' in label else 'LUAD' if 'LUAD' in label else 'Other' for label in labels])

# Create a mapping from label to a numerical value
label_to_color = {'LSCC': 0, 'LUAD': 1, 'Other': 2}
colors = np.array([label_to_color[label] for label in simplified_labels])

# 4. Dimensionality Reduction with t-SNE for Visualization
tsne = TSNE(n_components=2, random_state=42)
features_2d = tsne.fit_transform(features)

# Define the t-SNE Component 2 range for highlighting
tsne2_threshold = 40
tsne1_threshold = -60

# Filter points with t-SNE Component 2 > 60
highlight_indices = (features_2d[:, 1] < tsne2_threshold) & (features_2d[:, 1] < tsne1_threshold)

# Get the corresponding true labels for the highlighted points
highlighted_labels = simplified_labels[highlight_indices]

# Count the number of points for each label in the highlighted area
label_counts = Counter(highlighted_labels)
print("Label distribution where t-SNE Component 2 > 60:")
for label, count in label_counts.items():
    print(f"{label}: {count} points")

# 5. Plot the Features
plt.figure(figsize=(10, 7))

# Use the 'coolwarm' or 'bwr' colormap for blue and orange colors
scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=colors, cmap='coolwarm', marker='o')

plt.legend(handles=scatter.legend_elements()[0], labels=['LSCC', 'LUAD', 'Other'], title="Classes")
plt.title('t-SNE Visualization of Extracted Features')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.grid(False)

# Overlay images on t-SNE space in the highlighted area
for idx in np.where(highlight_indices)[0]:
    if idx >= len(images_list):
        print(f"Warning: Index {idx} is out of range for images_list with length {len(images_list)}.")
        continue
    img = images_list[idx]
    x, y = features_2d[idx, 0], features_2d[idx, 1]
    imagebox = OffsetImage(img, zoom=0.1)
    ab = AnnotationBbox(imagebox, (x, y), frameon=False)
    plt.gca().add_artist(ab)

plt.show()

plt.show()

