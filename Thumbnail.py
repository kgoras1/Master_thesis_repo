#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 12:10:18 2024

@author: konstantinospapagoras
"""

import os
import openslide

def thumbnail_svs_images(input_directory, output_directory):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Get a list of SVS files in the input directory
    svs_files = [file for file in os.listdir(input_directory) if file.endswith('.svs')]

    # Process each SVS file
    for svs_file in svs_files:
        try:
            # Open the SVS image
            slide = openslide.open_slide(os.path.join(input_directory, svs_file))

            # Generate the thumbnail image (1/32 of the original size)
            thumbnail = slide.get_thumbnail(size=(1200,1200))
                

            # Save the thumbnail image as JPEG or PNG in the output directory
            thumbnail_file = os.path.splitext(svs_file)[0] + '.jpg'# You can change the extension to '.png' if you want PNG format
            thumbnail_path = os.path.join(output_directory, thumbnail_file)
            thumbnail.save(thumbnail_path)

            print(f"Thumbnail saved: {thumbnail_path}")

        except Exception as e:
            print(f"Error processing {svs_file}: {e}")

if __name__ == "__main__":
    input_directory = input("Please provide the directory path containing SVS images: ")
    output_directory = "thumbnailed_LUCC"

    thumbnail_svs_images(input_directory, output_directory)
    
    print("The process is completed")
    
    
