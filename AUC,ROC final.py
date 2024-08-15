#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 11:48:29 2024

@author: konstantinospapagoras
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score

# Load the Excel file into a DataFrame
excel_file_path = '/Users/konstantinospapagoras/Master_Thesis/image_features.xlsx'  # Adjust the path if necessary
df = pd.read_excel(excel_file_path)

# Extract the true labels and predicted values
y_true = df['labels']
y_mean_values = df['mean_values']
y_avg_background_values = df['avg_background_values']

# First ROC Curve: Based on mean values
fpr_mean, tpr_mean, thresholds_mean_sklearn = roc_curve(y_true, y_mean_values)
roc_auc_mean = auc(fpr_mean, tpr_mean)

# Calculate Youden's J statistic for each threshold
youden_j_mean = tpr_mean - fpr_mean
optimal_idx_mean = np.argmax(youden_j_mean)
optimal_threshold_mean = thresholds_mean_sklearn[optimal_idx_mean]
optimal_threshold_mean_scaled = optimal_threshold_mean * 255

# Second ROC Curve: Based on average background values
fpr_bgr, tpr_bgr, thresholds_bgr_sklearn = roc_curve(y_true, y_avg_background_values)
roc_auc_bgr = auc(fpr_bgr, tpr_bgr)

# Calculate Youden's J statistic for each threshold
youden_j_bgr = tpr_bgr - fpr_bgr
optimal_idx_bgr = np.argmax(youden_j_bgr)
optimal_threshold_bgr = thresholds_bgr_sklearn[optimal_idx_bgr]
optimal_threshold_bgr_scaled = 100 - (optimal_threshold_bgr * 100)

# Third ROC Curve: Based on combined scores
combined_scores = np.where(y_mean_values >= optimal_threshold_mean, y_avg_background_values, 0)
fpr_combined, tpr_combined, thresholds_combined_sklearn = roc_curve(y_true, combined_scores)
roc_auc_combined = auc(fpr_combined, tpr_combined)

# Calculate Youden's J statistic for each threshold
youden_j_combined = tpr_combined - fpr_combined
optimal_idx_combined = np.argmax(youden_j_combined)
optimal_threshold_combined = thresholds_combined_sklearn[optimal_idx_combined]
optimal_threshold_combined_scaled = optimal_threshold_combined

# Plot ROC curve for mean values
plt.figure(figsize=(8, 6))
plt.plot(fpr_mean, tpr_mean, color='b', label=f'Mean Values (AUC = {roc_auc_mean:.2f})')
plt.scatter(fpr_mean[optimal_idx_mean], tpr_mean[optimal_idx_mean], color='g', label=f'Optimal Threshold (Mean) = {optimal_threshold_mean_scaled:.2f}')
plt.plot([0, 1], [0, 1], linestyle='-', color='r', alpha=0.5)
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve for Mean Values')
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.show()

# Plot ROC curve for average background values
plt.figure(figsize=(8, 6))
plt.plot(fpr_bgr, tpr_bgr, color='purple', label=f'Average Background (AUC = {roc_auc_bgr:.2f})')
plt.scatter(fpr_bgr[optimal_idx_bgr], tpr_bgr[optimal_idx_bgr], color='blue', label=f'Optimal Threshold (Avg BGR) = {optimal_threshold_bgr_scaled:.2f}')
plt.plot([0, 1], [0, 1], linestyle='-', color='r', alpha=0.5)
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve for Average Background Values')
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.show()

# Plot ROC curve for combined scores
plt.figure(figsize=(8, 6))
plt.plot(fpr_combined, tpr_combined, color='orange', label=f'Combined Scores (AUC = {roc_auc_combined:.2f})')
plt.scatter(fpr_combined[optimal_idx_combined], tpr_combined[optimal_idx_combined], color='red', label=f'Optimal Threshold (Combined) = {optimal_threshold_combined_scaled:.2f}')
plt.plot([0, 1], [0, 1], linestyle='-', color='r', alpha=0.5)
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve for Combined Scores')
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.show()

# Print optimal thresholds and AUC values
print(f'Optimal Threshold for Mean Values: {optimal_threshold_mean_scaled:.2f}')
print(f'AUC for Mean Values: {roc_auc_mean:.2f}')
print(f'Optimal Threshold for Avg Background Values: {optimal_threshold_bgr_scaled:.2f}')
print(f'AUC for Avg Background Values: {roc_auc_bgr:.2f}')
print(f'Optimal Threshold for Combined Scores: {optimal_threshold_combined_scaled:.2f}')
print(f'AUC for Combined Scores: {roc_auc_combined:.2f}')
