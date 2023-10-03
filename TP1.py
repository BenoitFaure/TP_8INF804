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
    image = cv2.imread(f"{images_folder}\{folder}\Mask.JPG")

    mask = image < 1
    image[mask] = 0
    image[np.bitwise_not(mask)] = 1

    return image

def load_other_images(folder: str) -> List[np.array]:

    full_path = f"{images_folder}\{folder}"
    all_files = os.listdir(full_path)
    filtered_files = [file for file in all_files if file.lower().endswith('.jpg') and file != 'Reference.JPG' and file != 'Mask.JPG']

    full_path += '\\'
    images = [cv2.imread(full_path + file) for file in filtered_files]

    return images

# ------- DEFINE TECHNIQUE -------

def make_bb(image, reference):

    # Get grayscale
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    
    # Take difference
    diff_image = cv2.absdiff(img, ref)

    # Threshold image
    _, thr_img = cv2.threshold(diff_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Dilate image
    kernel = np.ones((10, 10),np.uint8)
    # mortho_img = cv2.dilate(thr_img, kernel, iterations=5)
    # kernel = np.ones((2, 2),np.uint8)
    # mortho_img = cv2.morphologyEx(mortho_img, cv2.MORPH_BLACKHAT, kernel)
    # mortho_img = cv2.erode(thr_img, kernel, iterations=5)
    # mortho_img = cv2.morphologyEx(thr_img, cv2.MORPH_TOPHAT, kernel)
    mortho_img = cv2.erode(thr_img, kernel, iterations = 2)
    mortho_img = cv2.dilate(mortho_img, kernel, iterations=5)
    mortho_img = cv2.erode(mortho_img, kernel, iterations=3)


    # Find contours in the thresholded image
    contours, _ = cv2.findContours(mortho_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through the detected contours and draw bounding boxes
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return thr_img, image


mask = load_mask(folders[0])
mkbb = make_bb(load_other_images(folders[0])[0] * mask, load_reference_image(folders[0]) * mask)



cv2.imshow('make bb 1', cv2.resize(mkbb[0], (780, 540),
               interpolation = cv2.INTER_LINEAR))
cv2.imshow('make bb 2', cv2.resize(mkbb[1], (780, 540),
               interpolation = cv2.INTER_LINEAR))
cv2.waitKey(0)