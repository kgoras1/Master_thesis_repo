#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 16:08:42 2024

@author: konstantinospapagoras
"""

import csv

# Define file paths
csv_file1 = '/Users/konstantinospapagoras/Master_Thesis/Master_thesis_repo/LightXGBoost_Histo_classified_tiles.csv'
csv_file2 = '/Users/konstantinospapagoras/Master_Thesis/Master_thesis_repo/classified_tiles.csv'
output_txt_file = 'matched_tiles.txt'

# Function to read CSV and return a dictionary with tile names as keys and predictions as values
def read_csv_to_dict(file_path):
    predictions_dict = {}
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        for row in reader:
            tile_name, prediction = row
            predictions_dict[tile_name] = prediction
    return predictions_dict

# Read the predictions from both CSV files
predictions_dict1 = read_csv_to_dict(csv_file1)
predictions_dict2 = read_csv_to_dict(csv_file2)

# Find matching tiles and predictions
matching_tiles = []
for tile_name, prediction1 in predictions_dict1.items():
    if tile_name in predictions_dict2:
        prediction2 = predictions_dict2[tile_name]
        if prediction1 == prediction2:
            matching_tiles.append((tile_name, prediction1))

# Write the matching tiles to a text file
with open(output_txt_file, mode='w') as file:
    for tile_name, prediction in matching_tiles:
        file.write(f"{tile_name}, {prediction}\n")

print(f"Matching tiles and predictions saved to {output_txt_file}")
