from typing import List

import os
import re

import cv2
import numpy as np

# ------- IMAGE PROCESS -------
images_folder = '.\images'
folders = ['Chambre', 'Cuisine', 'Salon']

def load_reference_image(folder: str) -> np.array:
    return cv2.imread(f"{images_folder}\{folder}\Reference.jpg")

def load_mask(folder: str) -> np.array:
    image = cv2.imread(f"{images_folder}\{folder}\Mask.jpg", cv2.IMREAD_GRAYSCALE)

    image[image < 125] = 0
    image[image > 0] = 1

    return image

def load_other_images(folder: str) -> List[np.array]:

    full_path = f"{images_folder}\{folder}"
    all_files = os.listdir(full_path)
    filtered_files = [file for file in all_files if file.lower().endswith('.jpg') and file != 'Reference.JPG' and file != 'Mask.JPG']

    full_path += '\\'
    images = [cv2.imread(full_path + file) for file in filtered_files]

    return images

