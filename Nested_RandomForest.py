#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 15:08:29 2024

@author: konstantinospapagoras
"""

import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, matthews_corrcoef, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Constants
NUM_CLASSES = 2
OUTER_TEST_RATIO = 0.1  # 90/10 split for the outer loop
INNER_FOLDS = 6  # Number of inner folds

# Load pre-extracted VGG16 features and labels from pickle
with open('/Volumes/Seagate Expansion Drive/HistoEncoder/Histoencoder_features_labels.final.pkl', 'rb') as file:
    features, labels = pickle.load(file)

# Convert lists to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Get unique slide names and corresponding labels
unique_slides = np.array(sorted(list(set(labels))))
slide_labels = np.array([0 if 'LSCC' in slide else 1 for slide in unique_slides])

# Function for majority voting with vote statistics
def majority_voting(predictions):
    vote_counts = np.bincount(predictions, minlength=NUM_CLASSES)
    majority_vote = np.argmax(vote_counts)
    return majority_vote, vote_counts

def calculate_metrics(y_true, y_pred, y_pred_proba):
    auc_score = roc_auc_score(y_true, y_pred_proba[:, 1])
    mcc = matthews_corrcoef(y_true, y_pred)
    return auc_score, mcc

# Split the data for the outer loop manually with a different random state
outer_train_slides, outer_test_slides, outer_train_labels, outer_test_labels = train_test_split(
    unique_slides, slide_labels, test_size=OUTER_TEST_RATIO, stratify=slide_labels, random_state=94)

print("Outer Train Slides:", outer_train_slides)
print("Outer Test Slides:", outer_test_slides)

# Inner cross-validation within the outer training set
all_inner_train_errors = []
all_inner_val_errors = []
all_inner_auc = []
all_inner_mcc = []

# For ROC curve
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

# Define inner folds manually
# Create a mask to split slides into classes
lscc_slides = outer_train_slides[outer_train_labels == 0]
luad_slides = outer_train_slides[outer_train_labels == 1]

# Ensure we have enough slides for this setup
assert len(lscc_slides) >= INNER_FOLDS and len(luad_slides) >= INNER_FOLDS, "Not enough slides to create unique validation sets."

# Create the splits
for inner_fold in range(INNER_FOLDS):
    # Manually pick validation slides (1 from each class)
    val_lscc_slide = lscc_slides[inner_fold % len(lscc_slides)]
    val_luad_slide = luad_slides[inner_fold % len(luad_slides)]
    val_slides = np.array([val_lscc_slide, val_luad_slide])
    
    # Remaining slides for training
    train_slides = np.setdiff1d(outer_train_slides, val_slides)
    
    # Get corresponding labels
    val_labels = np.array([0, 1])  # LSCC -> 0, LUAD -> 1
    train_labels = slide_labels[np.isin(unique_slides, train_slides)]
    
    print(f"  Inner Fold {inner_fold+1}/{INNER_FOLDS}")
    print("  Inner Train Slides:", train_slides[:10])  # Print only first 10
    print("  Validation Slides:", val_slides)

    # Ensure 10 slides for training
    train_slides = train_slides[:10]  # Adjust to ensure only 10 slides used

    # Extract training features and labels
    inner_train_features, inner_train_labels = [], []
    for slide in train_slides:
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
    model = RandomForestClassifier(n_estimators=100, random_state=42)  # Adjust the number of estimators as needed
    model.fit(inner_train_features, inner_train_labels)

    # Evaluate on validation set
    val_predictions = model.predict(val_features)
    val_predictions_proba = model.predict_proba(val_features)

    val_loss = np.mean(val_predictions != val_labels)
    auc_score, mcc = calculate_metrics(val_labels, val_predictions, val_predictions_proba)
    
    all_inner_val_errors.append(val_loss)
    all_inner_auc.append(auc_score)
    all_inner_mcc.append(mcc)
    
    train_predictions = model.predict(inner_train_features)
    train_loss = np.mean(train_predictions != inner_train_labels)
    all_inner_train_errors.append(train_loss)
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(val_labels, val_predictions_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)

    # Interpolate the TPR values for a common FPR grid
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0  # Ensure the curve starts at (0, 0)

    plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'ROC fold {inner_fold+1} (AUC = {roc_auc:.2f})')
    
    # Debug outputs for inner fold performance
    print(f"  Inner Fold {inner_fold+1} AUC: {auc_score:.2f}")
    print(f"  Inner Fold {inner_fold+1} MCC: {mcc:.2f}")

# Plot the mean ROC curve
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0  # Ensure the curve ends at (1, 1)
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)

plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})', lw=2, alpha=0.8)

# Plotting the variance as a shaded area
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2, label='± 1 std. dev.')

# Plot settings
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=0.8)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Inner Folds of  RF/HistoEncoder')
plt.legend(loc='lower right')
plt.show()

# Inner fold statistics
mean_inner_auc = np.mean(all_inner_auc)
mean_inner_mcc = np.mean(all_inner_mcc)

print(f"\nInner Fold Average AUC: {mean_inner_auc:.2f}")
print(f"Inner Fold Average MCC: {mean_inner_mcc:.2f}")
print(f"Inner fold validation error: {[round(err, 2) for err in all_inner_val_errors]}")

# Outer loop evaluation at the WSI level
outer_test_true_labels = []
outer_test_predicted_labels = []
outer_predicted_prob = []

# Store all tile-level labels and predictions for MCC calculation
all_tile_true_labels = []
all_tile_pred_labels = []

for slide in outer_test_slides:
    slide_mask = labels == slide
    slide_features = features[slide_mask]
    slide_labels = np.array([0 if 'LSCC' in slide else 1] * sum(slide_mask))

    predictions = model.predict(slide_features)
    predictions_proba = model.predict_proba(slide_features)

    # Store tile-level predictions and true labels
    all_tile_true_labels.extend(slide_labels)
    all_tile_pred_labels.extend(predictions)

    predicted_wsi_label, vote_counts = majority_voting(predictions)
    true_wsi_label = slide_labels[0]

    print(f"\nSlide: {slide}")
    print(f"  True WSI Label: {true_wsi_label}")
    print(f"  Predicted WSI Label: {predicted_wsi_label}")
    print(f"  Votes per class: {vote_counts}")

    outer_test_true_labels.append(true_wsi_label)
    outer_test_predicted_labels.append(predicted_wsi_label)
    outer_predicted_prob.append(predictions_proba)

# Calculate statistics for outer test results
outer_test_true_labels = np.array(outer_test_true_labels)
outer_test_predicted_labels = np.array(outer_test_predicted_labels)

# Calculate AUC using predicted labels instead of probabilities
outer_test_auc = roc_auc_score(outer_test_true_labels, outer_test_predicted_labels)

# Calculate tile-level MCC
all_tile_true_labels = np.array(all_tile_true_labels)
all_tile_pred_labels = np.array(all_tile_pred_labels)
outer_tile_mcc = matthews_corrcoef(all_tile_true_labels, all_tile_pred_labels)

# Print results
print(f"\nWSI-level Accuracy: {np.mean(outer_test_predicted_labels == outer_test_true_labels):.2f}")
print(f"Outer Test AUC: {outer_test_auc:.2f}")
print(f"Outer Tile-level MCC: {outer_tile_mcc:.2f}")
