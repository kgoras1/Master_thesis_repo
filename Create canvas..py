#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 11:22:44 2024

@author: konstantinospapagoras
"""
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os
import tempfile
import traceback

# Function to load images from a directory
def load_images_from_directory(directory):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):  # Add other image formats if needed
            images.append(os.path.join(directory, filename))
    return images

# Function to load images
def load_images(paths):
    images = []
    for path in paths:
        try:
            # Check if file extension is valid
            valid_extensions = ('.jpg', '.jpeg', '.png', '.gif')
            if os.path.splitext(path)[1].lower() not in valid_extensions:
                print(f"Skipping {path}: Invalid file format")
                continue
            
            image = Image.open(path)
            if image is not None:  # Check if the image is successfully loaded
                images.append(image)
            else:
                print(f"Unable to load image: {path}")
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            traceback.print_exc()  # Print traceback for detailed error information
    return images

# Function to create collage
def create_collage(images, output_file):
    # Initialize PDF canvas
    c = canvas.Canvas(output_file, pagesize=letter)
    width, height = letter

    # Define parameters for arranging images
    columns = 10
    rows = 10
    cell_width = width / columns
    cell_height = height / rows

    # Place images on canvas
    x, y = 0, height
    for image in images:
        if image is None:
            continue
        # Resize image to fit cell
        image.thumbnail((cell_width, cell_height))

        # Save image to a temporary file
        temp_image_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        image.save(temp_image_file.name)

        # Draw image onto canvas
        c.drawImage(temp_image_file.name, x, y - cell_height, width=image.width, height=image.height)

        # Delete temporary file
        temp_image_file.close()
        os.unlink(temp_image_file.name)

        # Move to next cell
        x += cell_width
        if x >= width:
            x = 0
            y -= cell_height
            if y <= 0:
                # Reset y and start a new page
                y = height
                c.showPage()
                c.save()
                print(f"PDF file saved as: {output_file}")  # Print output file path
                return output_file

    # Check if there are remaining images
    if len(images) % (columns * rows) != 0:
        c.showPage()
        c.save()
        print(f"PDF file saved as: {output_file}")  # Print output file path
        return output_file


if __name__ == "__main__":
    try:
        directory = input("Please provide the directory path containing images: ")
        
        # Load images from the directory
        image_paths = load_images_from_directory(directory)
        
        # Load images
        images = load_images(image_paths)

        # Output file name
        output_file = "collage.pdf"

        # Create collage and export as PDF
        create_collage(images, output_file)
        
        print(f"Collage file {output_file} created successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()  # Print traceback for detailed error information
