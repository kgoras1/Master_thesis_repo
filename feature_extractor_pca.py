#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 14:54:29 2024

@author: konstantinospapagoras
"""

import os
import time
import keras
import numpy as np
import matplotlib.pyplot as plt
import random 
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.models import Model
from sklearn.decomposition import PCA

model = keras.applications.VGG16(weights='imagenet', include_top=True)

def load_image(path):
    img = image.load_img(path, target_size=model.input_shape[1:3])
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x

feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)
feat_extractor.summary()

# img, x = load_image("/Users/konstantinospapagoras/Master_Thesis/pca/8_31.jpeg")
# feat = feat_extractor.predict(x)

# plt.figure(figsize=(16,4))
# plt.plot(feat[0])

images_path = '/Users/konstantinospapagoras/Master_Thesis/tiles.'
image_extensions = ['.jpg', '.png', '.jpeg']   # case-insensitive (upper/lower doesn't matter)


images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(images_path) for f in filenames if os.path.splitext(f)[1].lower() in image_extensions]


# Print the number of images to analyze
print("keeping %d images to analyze" % len(images))


tic = time.process_time()

features = []
for i, images_path in enumerate(images):
    if i % 500 == 0:
        toc = time.process_time()
        elapsed_time = toc - tic
        print("Analyzing image %d / %d. Time: %4.4f seconds." % (i, len(images), elapsed_time))
        tic = time.process_time()
    # Assuming load_image is a function that loads an image from its path
    img, x = load_image(images_path)
    feat = feat_extractor.predict(x)[0]  # Assuming predict() extracts features from the image
    features.append(feat)

print('Finished extracting features for %d images' % len(images))


features = np.array(features)

pca = PCA()
pca.fit(features)
 
 # Determine number of components to retain 95% variance
desired_variance_retained = 0.95 
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
num_components = np.argmax(cumulative_variance_ratio >= desired_variance_retained)
print("Number of principal components to retain {} of variance: {}".format(desired_variance_retained * 100, num_components))
 
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o', linestyle='-')
plt.title('Cumulative Explained Variance Ratio')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.axhline(y=desired_variance_retained, color='r', linestyle='--', label='95% Variance')
plt.legend()
plt.show()
 
pca = PCA(n_components=228)
pca.fit(features)
pca_features = pca.transform(features)

print("Original features shape:", features.shape)
print("Transformed features shape:", pca_features.shape)


plt.clf()  # Close any existing figures
plt.figure(figsize=(8, 6))  # Create a new figure
plt.title('Scatter Plot of Samples in PCA Space')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Adjust limits
plt.xlim(pca_features[:, 0].min() - 0.1, pca_features[:, 0].max() + 0.1)
plt.ylim(pca_features[:, 1].min() - 0.1, pca_features[:, 1].max() + 0.1)

plt.scatter(pca_features[:, 0], pca_features[:, 1], alpha=0.5)
plt.show()


