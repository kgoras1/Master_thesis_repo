#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 10:02:41 2024

@author: konstantinospapagoras
"""
import os
import numpy as np
from PIL import Image

def find_mean_std_pixel_value(root_dir):
    avg_pixel_value = []
    stddev_pixel_value = []

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".jpg") or file.lower().endswith(".jpeg"):
                image_path = os.path.join(root, file)
                image = np.array(Image.open(image_path))
                avg = np.mean(image)
                std = np.std(image)
                avg_pixel_value.append(avg)
                stddev_pixel_value.append(std)

    avg_pixel_value = np.array(avg_pixel_value)
    stddev_pixel_value = np.array(stddev_pixel_value)

    if avg_pixel_value.size > 0:
        print(f"Average pixel value for all images is: {avg_pixel_value.mean():.2f}")
    else:
        print("No images found in the directory.")

    if stddev_pixel_value.size > 0:
        print(f"Average std dev of pixel value for all images is: {stddev_pixel_value.mean():.2f}")
    else:
        print("No images found in the directory.")

    return avg_pixel_value, stddev_pixel_value

# Example usage
root_dir = "/Users/konstantinospapagoras/Master_Thesis/good/"
avg_pixel_value, stddev_pixel_value = find_mean_std_pixel_value(root_dir)


# Average pixel value for all blanck images is: 243.8498996125116
# Average std dev of pixel value for all blanck images is: 1.020217572043789

# Average pixel value for all partial images is: 223.12
# Average std dev of pixel value for all partial images is: 40.01


# Average pixel value for all  good images is: 172.83
# Average std dev of pixel value for all good images is: 48.15