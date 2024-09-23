#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 11:28:43 2024

@author: konstantinospapagoras
"""

import os
import numpy as np
import pickle
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, matthews_corrcoef, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from statsmodels.stats.contingency_tables import mcnemar

# Constants
NUM_CLASSES = 2
OUTER_TEST_RATIO = 0.1  # 90/10 split for the outer loop
INNER_FOLDS = 6  # Number of inner folds

# Start timer
start_time = time.time()

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

# Function to create the ANN model
def create_ann_model(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

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

# Collect predictions for ANN and LightGBM for McNemar's test on outer validation set
ann_outer_val_predictions = []
lgbm_outer_val_predictions = []

# For ROC curve
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

# Define inner folds manually
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

    # Extract training features and labels for ANN and LightGBM
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

    # Train ANN model
    ann_model = create_ann_model(input_dim=inner_train_features.shape[1])
    ann_model.fit(inner_train_features, inner_train_labels, epochs=10, batch_size=32, verbose=0)

    # Model training with LightGBM
    lgbm_model = lgb.LGBMClassifier(n_estimators=100, random_state=42)  # Adjust the parameters as needed
    lgbm_model.fit(inner_train_features, inner_train_labels)

    # Evaluate ANN on validation set
    ann_val_predictions_proba = ann_model.predict(val_features)
    ann_val_predictions = np.argmax(ann_val_predictions_proba, axis=1)
    ann_outer_val_predictions.append(ann_val_predictions)

    # Evaluate LightGBM on validation set
    lgbm_val_predictions_proba = lgbm_model.predict_proba(val_features)
    lgbm_val_predictions = np.argmax(lgbm_val_predictions_proba, axis=1)
    lgbm_outer_val_predictions.append(lgbm_val_predictions)

    # Calculate metrics for both models
    ann_auc_score, ann_mcc = calculate_metrics(val_labels, ann_val_predictions, ann_val_predictions_proba)
    lgbm_auc_score, lgbm_mcc = calculate_metrics(val_labels, lgbm_val_predictions, lgbm_val_predictions_proba)

    all_inner_auc.append((ann_auc_score, lgbm_auc_score))
    all_inner_mcc.append((ann_mcc, lgbm_mcc))
    
    print(f"  Inner Fold {inner_fold+1} ANN AUC: {ann_auc_score:.2f}, LightGBM AUC: {lgbm_auc_score:.2f}")
    print(f"  Inner Fold {inner_fold+1} ANN MCC: {ann_mcc:.2f}, LightGBM MCC: {lgbm_mcc:.2f}")

# Outer loop evaluation (for McNemar's test)

# Outer loop evaluation (for McNemar's test)

# Convert predictions lists to arrays for outer validation set
ann_outer_val_predictions = np.concatenate(ann_outer_val_predictions)
lgbm_outer_val_predictions = np.concatenate(lgbm_outer_val_predictions)
outer_true_labels = np.concatenate([val_labels] * INNER_FOLDS)  # True labels from validation sets

# Prepare contingency table for McNemar's test (2x2 table)
contingency_table = np.zeros((2, 2))

for true_label, ann_pred, lgbm_pred in zip(outer_true_labels, ann_outer_val_predictions, lgbm_outer_val_predictions):
    if ann_pred == true_label and lgbm_pred == true_label:
        contingency_table[0, 0] += 1  # Both models correct
    elif ann_pred == true_label and lgbm_pred != true_label:
        contingency_table[0, 1] += 1  # ANN correct, LightGBM incorrect
    elif ann_pred != true_label and lgbm_pred == true_label:
        contingency_table[1, 0] += 1  # ANN incorrect, LightGBM correct
    else:
        contingency_table[1, 1] += 1  # Both models incorrect

print(f"Contingency Table (based on comparison to true labels):\n{contingency_table}")

# Perform McNemar's test
result = mcnemar(contingency_table, exact=True)

print(f"McNemar's test statistic: {result.statistic}")
print(f"p-value: {result.pvalue}")

if result.pvalue < 0.05:
    print("Significant difference between ANN and LightGBM")
else:
    print("No significant difference between ANN and LightGBM")


# Plot the mean ROC curve for both models
mean_inner_ann_auc = np.mean([x[0] for x in all_inner_auc])
mean_inner_lgbm_auc = np.mean([x[1] for x in all_inner_auc])
print(f"\nInner Fold Average ANN AUC: {mean_inner_ann_auc:.2f}")
print(f"Inner Fold Average LightGBM AUC: {mean_inner_lgbm_auc:.2f}")

end_time = time.time()
print(f"Time taken for model training: {end_time - start_time:.2f} seconds")
