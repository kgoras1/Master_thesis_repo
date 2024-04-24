#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 18:21:36 2024

@author: konstantinospapagoras
"""

import os
import math
from openslide import OpenSlide
from PIL import Image
import glob
import numpy as np


def generate_deepzoom_tiles(svs_file, output_dir, tile_size=512, overlap=0):
    slide = OpenSlide(svs_file)
    
    level_count = slide.level_count
    level_tile_count = {}
    for level in range(level_count):
        scale = slide.level_downsamples[level]
        level_dir = os.path.join(output_dir, str(level))
        os.makedirs(level_dir, exist_ok=True)

        width, height = slide.level_dimensions[level]
        tile_width = tile_height = tile_size
        num_tiles_x = int(math.ceil(width / tile_width))
        num_tiles_y = int(math.ceil(height / tile_height))
        level_tile_count[level] = num_tiles_x * num_tiles_y

        for tile_x in range(num_tiles_x):
            for tile_y in range(num_tiles_y):
                tile_left = tile_x * tile_width
                tile_top = tile_y * tile_height
                tile_right = min(tile_left + tile_width + overlap, width)
                tile_bottom = min(tile_top + tile_height + overlap, height)

                tile = slide.read_region((tile_left, tile_top), level, (tile_right - tile_left, tile_bottom - tile_top))
                tile = tile.convert("RGB")
                tile_np = np.array(tile)
                if tile_np.mean() < 230 and tile_np.std() > 20 :

                    tile_filename = f"tile_{tile_x}_{tile_y}.jpg"
                    tile_path = os.path.join(level_dir, tile_filename)
                    tile.save(tile_path, "JPEG")
    
                    print(f"Saved tile: {tile_path}, Level: {level}, Coordinates: ({tile_left}, {tile_top})")

    slide.close()
    print("Tiles are generated and the process is completed")
    
    with open(os.path.join(output_dir,'tile_count.txt'),'w') as f:
        for level, count in level_tile_count.items():
            f.write(f"Level {level} : {count} tiles\n")
    print("The file is written")
    
            
            
# Example usage
svs_file = input("please provide with the svs file ")
output_dir = os.path.join(os.path.dirname(svs_file), "tiles")

# Call the function with user-provided input
generate_deepzoom_tiles(svs_file, output_dir)


