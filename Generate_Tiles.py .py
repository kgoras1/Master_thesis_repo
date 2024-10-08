#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 09:57:12 2024

@author: konstantinospapagoras
"""
import os
import math
import numpy as np
import cv2
import csv
from openslide import OpenSlide, OpenSlideError
from PIL import Image

def generate_deepzoom_tiles_from_path(input_path, output_dir, tile_size=512, overlap=0):
    """
    Generate deep zoom tiles from a directory or a single SVS file.

    Args:
    - input_path (str): Path to the directory or SVS file.
    - output_dir (str): Directory to save the tiles.
    - tile_size (int, optional): Size of the tiles. Default is 512.
    - overlap (int, optional): Overlap of the tiles. Default is 0.
    """
    if os.path.isdir(input_path):
        for root, _, files in os.walk(input_path):
            for file in files:
                if file.endswith(".svs"):
                    svs_file = os.path.join(root, file)
                    generate_deepzoom_tiles(svs_file, output_dir, tile_size=tile_size, overlap=overlap)
    elif os.path.isfile(input_path) and input_path.endswith(".svs"):
        generate_deepzoom_tiles(input_path, output_dir, tile_size=tile_size, overlap=overlap)
    else:
        print("The input path is neither a directory nor an SVS file.")

def generate_deepzoom_tiles(svs_file, output_dir, tile_size=512, overlap=0):
    """
    Generate deep zoom tiles from a single SVS file.

    Args:
    - svs_file (str): Path to the SVS file.
    - output_dir (str): Directory to save the tiles.
    - tile_size (int, optional): Size of the tiles. Default is 512.
    - overlap (int, optional): Overlap of the tiles. Default is 0.
    """
    try:
        slide = OpenSlide(svs_file)
    except OpenSlideError as e:
        print(f"Error opening slide file {svs_file}: {e}")
        return

    level_count = slide.level_count

    # Prepare CSV file for writing mean and average values
    csv_file_path = os.path.join(output_dir, 'tile_statistics.csv')
    with open(csv_file_path, mode='w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Writing the header row
        csvwriter.writerow(['Tile_X', 'Tile_Y', 'Level', 'Mean', 'Average'])

        for level in range(level_count):
            level_dir = os.path.join(output_dir, str(level))
            os.makedirs(level_dir, exist_ok=True)

            width, height = slide.level_dimensions[level]
            tile_width = tile_height = tile_size
            num_tiles_x = int(math.ceil(width / tile_width))
            num_tiles_y = int(math.ceil(height / tile_height))

            for tile_x in range(num_tiles_x):
                for tile_y in range(num_tiles_y):
                    tile_left = tile_x * tile_width
                    tile_top = tile_y * tile_height
                    tile_right = min(tile_left + tile_width + overlap, width)
                    tile_bottom = min(tile_top + tile_height + overlap, height)

                    tile = slide.read_region((tile_left, tile_top), level, (tile_right - tile_left, tile_bottom - tile_top))
                    tile = tile.convert("RGB")
                    tile_np = np.array(tile)

                    # Convert to grayscale
                    gray = tile.convert('L')
                    img_grey = cv2.cvtColor(tile_np, cv2.COLOR_BGR2GRAY)

                    # Create binary image
                    bw = gray.point(lambda x: 0 if x < 230 else 1, 'F')
                    arr = np.array(bw)
                    avgBkg = np.average(arr)

                    # Calculate mean of the tile
                    tile_mean = tile_np.mean()

                    # Save the mean and avgBkg values to the CSV file
                    csvwriter.writerow([tile_x, tile_y, level, tile_mean, avgBkg])

                    print(f"Processed tile: Level {level}, Coordinates ({tile_left}, {tile_top}) - Mean: {tile_mean}, Average: {avgBkg}")

    slide.close()
    print("Tiles processed and statistics saved to CSV.")

# Example usage
input_dir = input("Please provide the input directory: ")
output_dir = os.path.join(os.path.dirname(input_dir), "LUAD_7_512nobgr")

# Call the function with user-provided input
generate_deepzoom_tiles_from_path(input_dir, output_dir)
