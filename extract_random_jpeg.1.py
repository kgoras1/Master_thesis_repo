#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 17:01:00 2024

@author: konstantinospapagoras
"""

import os
import random
import shutil

import os
import random
import shutil

def extract_random_jpeg(input_directory, output_directory):
    # Check if the input directory exists
    if not os.path.exists(input_directory):
        print(f"Input directory '{input_directory}' not found.")
        return

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Recursively search for subdirectories
    for root, dirs, files in os.walk(input_directory):
        for subdir in dirs:
            subdir_path = os.path.join(root, subdir)
            # Get a list of JPEG files in the subdirectory
            jpeg_files = [file for file in os.listdir(subdir_path) if file.lower().endswith('.jpg') or file.lower().endswith('.jpeg')]
            # If there are JPEG files, select a random one and copy it to the output directory maintaining the directory structure
            if jpeg_files:
                random_jpeg = random.choice(jpeg_files)
                source_file = os.path.join(subdir_path, random_jpeg)
                destination_file = os.path.join(output_directory, random_jpeg)  # Specify full destination file path
                shutil.copyfile(source_file, destination_file)
                print(f"Random JPEG file '{random_jpeg}' from '{subdir_path}' copied to '{destination_file}'")
            else:
                print(f"No JPEG files found in '{subdir_path}'")

# Ask for the input and output directories
input_directory = input("Enter the input directory path: ")
output_directory = input("Enter the output directory path: ")

# Extract and save random JPEG files from subdirectories
extract_random_jpeg(input_directory, output_directory)

