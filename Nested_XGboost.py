#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 10:12:34 2024

@author: konstantinospapagoras
"""

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, matthews_corrcoef
import xgboost as xgb
from scipy.stats import mode
import random

# Constants
NUM_CLASSES = 2
OUTER_FOLDS = 1  # Set to 1 to ensure one specific split
INNER_FOLDS = 5  # Set to 3 for the inner cross-validation

# Load pre-extracted VGG16 features and labels from pickle
with open('/Volumes/Seagate Expansion Drive/HistoEncoder/512features_labels_Histoencoder.pkl', 'rb') as file:
    features, labels = pickle.load(file)

# Convert lists to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Get unique slide names
unique_slides = sorted(list(set(labels)))

# Function for majority voting
def majority_voting(predictions):
    majority_vote = mode(predictions, axis=None, keepdims=True)[0][0]
    return majority_vote

# Function to calculate AUC and MCC
def calculate_metrics(y_true, y_pred, y_pred_proba):
    auc = roc_auc_score(y_true, y_pred_proba[:, 1]) if len(np.unique(y_true)) > 1 else None
    mcc = matthews_corrcoef(y_true, y_pred)
    return auc, mcc

# Manually define one outer fold with specific conditions
lscc_slides = [slide for slide in unique_slides if 'LSCC' in slide]
luad_slides = [slide for slide in unique_slides if 'LUAD' in slide]

# Define the number of training and testing slides
num_train = 4
num_test = 1

# Randomly select training indices
all_indices_lscc = list(range(len(lscc_slides)))
print(all_indices_lscc)
all_indices_luad = list(range(len(luad_slides)))
print(all_indices_luad)
train_indices_lscc = random.sample(all_indices_lscc, num_train)
train_indices_luad = random.sample(all_indices_luad, num_train)

# Ensure that testing indices are not in the training set
test_indices_lscc = list(set(all_indices_lscc) - set(train_indices_lscc))[:num_test]
test_indices_luad = list(set(all_indices_luad) - set(train_indices_luad))[:num_test]

# Prepare training and testing slides
outer_train_slides = [lscc_slides[i] for i in train_indices_lscc] + [luad_slides[i] for i in train_indices_luad]
outer_test_slides = [lscc_slides[i] for i in test_indices_lscc] + [luad_slides[i] for i in test_indices_luad]

print("Outer Train Slides:", outer_train_slides)
print("Outer Test Slides:", outer_test_slides)

# Inner cross-validation with 3 folds
inner_kf = KFold(n_splits=INNER_FOLDS, shuffle=True, random_state=42)
all_inner_train_errors = []
all_inner_val_errors = []
all_inner_auc = []
all_inner_mcc = []

for inner_fold, (inner_train_index, val_index) in enumerate(inner_kf.split(outer_train_slides)):
    print(f"  Inner Fold {inner_fold+1}/{INNER_FOLDS}")
    
    inner_train_slides = [outer_train_slides[i] for i in inner_train_index]
    val_slides = [outer_train_slides[i] for i in val_index]
    
    print("  Inner Train Slides:", inner_train_slides)
    print("  Validation Slides:", val_slides)

    # Extract training features and labels
    inner_train_features, inner_train_labels = [], []
    for slide in inner_train_slides:
        slide_mask = labels == slide
        inner_train_features.append(features[slide_mask])
        inner_train_labels.append(np.array([0 if 'LSCC' in slide else 1] * sum(slide_mask)))
    inner_train_features = np.concatenate(inner_train_features)
    inner_train_labels = np.concatenate(inner_train_labels)

    val_features, val_labels = [], []
    for slide in val_slides:
        slide_mask = labels == slide
        val_features.append(features[slide_mask])
        val_labels.append(np.array([0 if 'LSCC' in slide else 1] * sum(slide_mask)))
    val_features = np.concatenate(val_features)
    val_labels = np.concatenate(val_labels)

    # Model training
    model = xgb.XGBClassifier(eval_metric='logloss')
    model.fit(inner_train_features, inner_train_labels)
    
    # Evaluate on validation set
    val_predictions = model.predict(val_features)
    val_predictions_proba = model.predict_proba(val_features)
    
    val_loss = np.mean(val_predictions != val_labels)
    auc, mcc = calculate_metrics(val_labels, val_predictions, val_predictions_proba)
    
    all_inner_val_errors.append(val_loss)
    all_inner_auc.append(auc)
    all_inner_mcc.append(mcc)
    
    # Track train error (if needed)
    train_predictions = model.predict(inner_train_features)
    train_loss = np.mean(train_predictions != inner_train_labels)
    all_inner_train_errors.append(train_loss)
    
# Evaluate on outer test set at WSI level
correct_predictions = 0
outer_test_auc = []
outer_test_mcc = []

for slide in outer_test_slides:
    slide_mask = labels == slide
    slide_features = features[slide_mask]
    slide_labels = np.array([0 if 'LSCC' in slide else 1] * sum(slide_mask))

    predictions = model.predict(slide_features)
    predictions_proba = model.predict_proba(slide_features)
    
    predicted_wsi_label = majority_voting(predictions)
    true_wsi_label = slide_labels[0]  # All tiles have the same label

    print(f"Slide {slide}:")
    print(f"  True WSI Label: {true_wsi_label}")
    print(f"  Predicted WSI Label: {predicted_wsi_label}")
    print(f"  Predictions: {predictions}")
    print(f"  Probabilities: {predictions_proba}")

    if predicted_wsi_label == true_wsi_label:
        correct_predictions += 1

    # Calculate metrics at WSI level
    auc, mcc = calculate_metrics(slide_labels, predictions, predictions_proba)
    outer_test_auc.append(auc)
    outer_test_mcc.append(mcc)
    
    # Optional: Track tile votes for statistics
    unique, counts = np.unique(predictions, return_counts=True)
    tile_vote_stats = dict(zip(unique, counts))
    print(f"  Votes per class: {tile_vote_stats}")

# Store the WSI-level accuracy for this outer fold
test_accuracy = correct_predictions / len(outer_test_slides)
print(f"WSI-level Accuracy: {test_accuracy}")
print(f"Outer Test AUC: {outer_test_auc}")
print(f"Outer Test MCC: {outer_test_mcc}")

# Plot inner training and validation errors (averaged over folds)
plt.figure(figsize=(10, 6))
if len(all_inner_train_errors) > 1:
    plt.plot(range(1, len(all_inner_train_errors) + 1), all_inner_train_errors, label='Inner Training Error')
if len(all_inner_val_errors) > 1:
    plt.plot(range(1, len(all_inner_val_errors) + 1), all_inner_val_errors, label='Inner Validation Error')

plt.xlabel('Fold')
plt.ylabel('Error Rate')
plt.title('Inner Training and Validation Errors')
plt.legend()
plt.show()

# Display AUC and MCC for inner folds
print(f"Inner Fold AUC: {all_inner_auc}")
print(f"Inner Fold MCC: {all_inner_mcc}")

