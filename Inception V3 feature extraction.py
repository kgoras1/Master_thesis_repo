#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 16:57:53 2024

@author: konstantinospapagoras
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
import pickle
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Initialize InceptionV3 model for feature extraction
model = InceptionV3(weights='imagenet', include_top=True)
feat_extractor = Model(inputs=model.input, outputs=model.get_layer("avg_pool").output)
print("\nFeature Extractor Model Summary:")
feat_extractor.summary()

def load_image(path):
    try:
        img = image.load_img(path, target_size=model.input_shape[1:3])
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
        feat = feat_extractor.predict(x)[0]
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

        # Extract features using InceptionV3 with progress bar and parallel processing
        with ThreadPoolExecutor() as executor:
            results = list(tqdm(executor.map(lambda img_path: process_image(img_path, first_subdir), images), total=len(images)))

        for feat, label, img in results:
            if feat is not None and label is not None and img is not None:
                features.append(feat)
                labels.append(label)
                images_list.append(img)  # Store the original images

# Convert lists to numpy arrays
features = np.array(features)
labels = np.array(labels)

print(len(labels))

# Save features and labels to a pickle file
pickle.dump((features, labels), open(os.path.join(input_dir, 'features_labels_InceptionV3.final.pkl'), 'wb'))
