#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 10:57:00 2024

@author: konstantinospapagoras
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
import pickle
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pickle
import seaborn as sns
from sklearn.metrics import adjusted_mutual_info_score

# Initialize VGG16 model for feature extraction
model = VGG16(weights='imagenet', include_top=True)
feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)
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
input_dir = '/Users/konstantinospapagoras/Desktop/data/'

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

        # Extract features using VGG16 with progress bar and parallel processing
        with ThreadPoolExecutor() as executor:
            results = list(tqdm(executor.map(process_image, images, [first_subdir]*len(images)), total=len(images)))

        for feat, label, img in results:
            if feat is not None and label is not None and img is not None:
                features.append(feat)
                labels.append(label)
                images_list.append(img)  # Store the original images

# Convert lists to numpy arrays
features = np.array(features)
labels = np.array(labels)

print(len(features))
# Save features and labels to a pickle file
pickle.dump((features, labels), open(os.path.join(input_dir, 'features_data_labels.pkl'), 'wb'))

# #Simplify the true labels
# simplified_labels = np.array(['LSCC' if 'LSCC' in label else 'LUAD' for label in labels])
# simplified_labels_u = np.unique(simplified_labels)

# # Print unique labels
# unique_labels = np.unique(labels)
# print(f"Unique labels: {unique_labels}")
# print(f"Simplified labels: {simplified_labels_u}")

# # Perform PCA to reduce dimensionality
# pca = PCA()
# pca.fit(features)

# # Determine number of components to retain 95% variance
# desired_variance_retained = 0.95
# cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
# num_components = np.argmax(cumulative_variance_ratio >= desired_variance_retained) + 1
# print(f"Number of principal components to retain {desired_variance_retained * 100}% of variance: {num_components}")

# # Perform PCA with the determined number of components
# pca = PCA(n_components=num_components)
# pca_components = pca.fit_transform(features)

# # Apply K-means clustering and compute silhouette scores for k values 2 to 5
# range_k = range(2, 11)
# silhouette_scores = []

# for k in range_k:
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     cluster_labels = kmeans.fit_predict(pca_components)
#     silhouette_avg = silhouette_score(pca_components, cluster_labels)
#     silhouette_scores.append(silhouette_avg)
#     print(f"For n_clusters = {k}, the average silhouette score is {silhouette_avg}")

# # Find the optimal k with the highest silhouette score
# optimal_k = range_k[np.argmax(silhouette_scores)]
# print(f"The optimal number of clusters is {optimal_k}")

# # Plot silhouette scores
# plt.figure(figsize=(10, 5))
# plt.plot(range_k, silhouette_scores, marker='o')
# plt.title('Silhouette Scores for K-means Clustering')
# plt.xlabel('Number of clusters (k)')
# plt.ylabel('Silhouette Score')
# plt.xticks(range_k)

# # Highlight the point with the best silhouette score with a red circle
# best_k = optimal_k
# best_score = silhouette_scores[best_k - 2]  # Index starts from 2
# plt.plot(best_k, best_score, marker='o', markersize=8, color='red')  # Highlight the best point
# plt.grid(False)  # Turn off the grid
# plt.show()

# # Apply K-means clustering with the optimal k
# kmeans = KMeans(n_clusters=2, random_state=42)
# cluster_labels = kmeans.fit_predict(features)

# # Create a DataFrame with true labels and cluster labels
# df = pd.DataFrame({'True Label': simplified_labels, 'Cluster Label': cluster_labels})

# # Create a contingency table (cross-tabulation) to count the occurrences of each combination of true and cluster labels
# contingency_table = pd.crosstab(df['True Label'], df['Cluster Label'], rownames=['True Label'], colnames=['Cluster Label'])

# # Display the flipped contingency table
# print(contingency_table)

# # Visualize the contingency table using a heatmap with purple color map
# plt.figure(figsize=(10, 8))
# sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Purples')
# plt.title('Contingency Table of Cluster Labels vs True Labels')
# plt.show()

# # Plot the PCA results with simplified true labels and clusters
# # Define a consistent color palette for plotting using seaborn palette
# true_label_order = np.unique(simplified_labels)
# cluster_label_order = np.unique(cluster_labels)

# # Use the same color palette for both sets of labels
# palette = sns.color_palette("tab10", len(true_label_order) + len(cluster_label_order))
# custom_palette_true = {label: palette[i] for i, label in enumerate(true_label_order)}
# custom_palette_cluster = {label: palette[len(true_label_order) + i] for i, label in enumerate(cluster_label_order)}

# # Create a scatter plot for true labels and clusters on the same PCA plot
# plt.figure(figsize=(14, 6))

# # Scatter plot for true labels
# plt.subplot(1, 2, 1)
# for label in true_label_order:
#     idx = simplified_labels == label
#     plt.scatter(pca_components[idx, 0], pca_components[idx, 1], color=custom_palette_true[label], label=label, alpha=0.5, marker='o')
# plt.title('PCA with Simplified True Labels')
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.legend()

# # Scatter plot for predicted clusters
# plt.subplot(1, 2, 2)
# for label in cluster_label_order:
#     idx = cluster_labels == label
#     plt.scatter(pca_components[idx, 0], pca_components[idx, 1], color=custom_palette_cluster[label], label=f'Cluster {label}', alpha=0.5, marker='x')
# plt.title('PCA with Cluster Predictions')
# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.legend()

# plt.show()

# # Calculate the mutual information between true labels and cluster labels
# mi_inception = adjusted_mutual_info_score(simplified_labels, cluster_labels)
# print(f"Mutual Information for VGG16 features: {mi_inception}")

# # Analyze mutual information for each cluster
# for cluster in cluster_label_order:
#     idx = cluster_labels == cluster
#     mi_cluster = adjusted_mutual_info_score(simplified_labels[idx], cluster_labels[idx])
#     print(f"Mutual Information for cluster {cluster}: {mi_cluster}")



