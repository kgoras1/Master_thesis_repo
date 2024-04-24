#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 10:41:39 2024

@author: konstantinospapagoras
"""

import os

def count_patients_WSIs(directory):
    patient_ID = {}
    total_images = 0
    total_size = 0
    
    for root, dirs, files in os.walk(directory):
        for svs in files:
            if svs.endswith('.svs'):
                patient_num = svs.split('-')[1]
                patient_label = svs.split('-')[0]
                
                # Assigning label based on patient label
                if patient_label == "C3L":
                    label = 'LSCC'
                else:
                    label = 'LUAD'
                
                # Increment image count for the patient
                if patient_num not in patient_ID:
                    patient_ID[patient_num] = {'label': label, 'count': 0}
                    
                patient_ID[patient_num]['count'] += 1
                patient_ID[patient_num]['count']
                # Increment total image count
                total_images += 1  
                
                # Get file path and add file size to total size
                file_path = os.path.join(root, svs)
                total_size += os.path.getsize(file_path)  
    
    num_patients = len(patient_ID)  # Count the number of patients
    
    return patient_ID, num_patients, total_images, total_size

def bytes_to_gb(size_in_bytes):
    return size_in_bytes / (1024 ** 3)

if __name__ == "__main__":
    directory = input("Please provide the directory path: ")
    patient_counts, num_patients, total_images, total_size = count_patients_WSIs(directory)
    total_size_gb = bytes_to_gb(total_size)
    
    # Save outputs to a text file
    with open("info_LCCS_dataset.txt", "w") as f:
        f.write("Patient Number\tLabel\tImage Count\n")
        for patient_number, info in patient_counts.items():
            f.write(f"{patient_number}\t\t{info['label']}\t\t{info['count']}\n")

        f.write("\nNumber of patients in the dataset: {}\n".format(num_patients))
        f.write("Total number of images: {}\n".format(total_images))
        f.write("Total size of SVS files: {:.2f} GB\n".format(total_size_gb))

    print("Outputs saved to info_LCCS_dataset.txt")
