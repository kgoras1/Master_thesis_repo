#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 09:11:27 2024

@author: konstantinospapagoras
"""

import os
import torch
import histoencoder.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pickle

# 1. Initialize the Encoder
encoder = F.create_encoder("prostate_small")  # Replace with your desired encoder
encoder.eval()  # Set to evaluation mode

# 2. Define the Image Preprocessing Steps
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust as necessary
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to load and preprocess a single image
def load_image(path):
    try:
        img = Image.open(path).convert('RGB')
        img = preprocess(img)
        img = img.unsqueeze(0)  # Add batch dimension
        return img
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None

# Define input directory containing the subdirectories
input_dir = '/Volumes/Seagate Expansion Drive/LSCC:LUAD:512NoBackroundFinal_Tiles/'  # Corrected the directory path

# Initialize lists to store data
features = []
labels = []
images_list = []

# 3. Extract Features from Images
for root, dirs, files in os.walk(input_dir):
    for dir_name in dirs:  # Iterate over subdirectories
        dir_path = os.path.join(root, dir_name)  # Path to the current subdirectory
        for file in os.listdir(dir_path):  # Iterate over files in the subdirectory
            if file.endswith(('jpg', 'jpeg', 'png')):
                image_path = os.path.join(dir_path, file)
                label = dir_name  # Use the directory name as the label

                img = load_image(image_path)
                if img is not None:
                    with torch.no_grad():  # Disable gradients for feature extraction
                        # Extract features using the encoder
                        feature = F.extract_features(encoder, img, num_blocks=2, avg_pool=True)

                    # Convert feature to numpy and store it
                    features.append(feature.cpu().numpy().flatten())
                    labels.append(label)
                    images_list.append(image_path)

# Convert features and labels to numpy arrays
features_np = np.array(features)
labels_np = np.array(labels)

print(f"Unique labels: {np.unique(labels_np)}")
print(f"Number of images: {len(images_list)}")
print(f"Example image paths: {images_list[:5]}")

output_file_path = '/Volumes/Seagate Expansion Drive/HistoEncoder/512features_labels_Histoencoder.pkl'
# Create the output directory if it doesn't exist
output_dir = os.path.dirname(output_file_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save features and labels to a pickle file
try:
    with open(output_file_path, 'wb') as f:
        pickle.dump((features_np, labels_np), f)
    print(f"Features and labels successfully saved to {output_file_path}")
except Exception as e:
    print(f"An error occurred: {e}")
