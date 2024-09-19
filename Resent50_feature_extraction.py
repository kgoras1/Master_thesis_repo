#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 08:29:30 2024

@author: konstantinospapagoras
"""

import os
import numpy as np
import pickle
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Initialize ResNet50 model for feature extraction
# Use include_top=False to exclude the final classification layers
model = ResNet50(weights='imagenet', include_top=False)
# Extract features from the 'conv5_block3_add' layer
feat_extractor = Model(inputs=model.input, outputs=model.get_layer("conv5_block3_add").output)
print("\nFeature Extractor Model Summary:")
feat_extractor.summary()

def load_image(path):
    try:
        # Load and preprocess the image
        img = image.load_img(path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return img, x
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None, None

# Define input directory containing the subdirectories
input_dir = '/Volumes/Seagate Expansion Drive/LSCC:LUAD:512NoBackroundFinal_Tiles/'

# Initialize lists to store data
features = []
labels = []
images_list = []

def process_image(img_path, first_subdir):
    img, x = load_image(img_path)
    if img is not None and x is not None:
        feat = feat_extractor.predict(x)
        feat = feat.flatten()  # Flatten the output to a 1D array
        return feat, first_subdir, img
    return None, None, None

# Traverse through the directory structure
for root, dirs, files in os.walk(input_dir):
    if files:  # If the current directory has files
        # Identify the first subdirectory this folder belongs to
        relative_path = os.path.relpath(root, input_dir)
        first_subdir = relative_path.split(os.sep)[0]

        # Process the images in the current directory
        images = [os.path.join(root, f) for f in files if f.lower().endswith(('jpg', 'jpeg', 'png'))]
        print(f"Processing {len(images)} images in {relative_path}...")

        # Extract features using ResNet50 with progress bar and parallel processing
        with ThreadPoolExecutor() as executor:
            results = list(tqdm(executor.map(process_image, images, [first_subdir] * len(images)), total=len(images)))

        for feat, label, img in results:
            if feat is not None and label is not None:
                features.append(feat)
                labels.append(label)
                images_list.append(img)  # Store the original images if needed

# Convert lists to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Ensure that the output is 2D: (number of samples, features per sample)
print(f"Features shape: {features.shape}")
print(f"Labels shape: {labels.shape}")

# Save features and labels to a pickle file
output_path = os.path.join(input_dir, 'resnet50_features_labels_final.pkl')
with open(output_path, 'wb') as f:
    pickle.dump((features, labels), f)

print(f"Features and labels saved to {output_path}")
