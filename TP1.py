from typing import List

import os
import re

import cv2
import numpy as np

import matplotlib.pyplot as plt

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

def make_bb(image, reference, mask):

    # Map the reference luminosity to the image
    img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)

    # img_h = cv2.calcHist([img_lab[:, :, 0]], [0], None, [256], [0, 256])
    # ref_h = cv2.calcHist([ref_lab[:, :, 0]], [0], None, [256], [0, 256])

    # img_cdf = img_h.cumsum()
    # ref_cdf = ref_h.cumsum()

    # img_cdf = img_cdf / img_cdf.max()
    # ref_cdf = ref_cdf / ref_cdf.max()

    # mapping = np.interp(img_cdf, ref_cdf, np.arange(256))

    # img_lab[:, :, 0] = ref_lab[:, :, 0] # mapping[img_lab[:, :, 0]]

    img = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
    ref = cv2.cvtColor(ref_lab, cv2.COLOR_LAB2BGR)

    a = img_lab[:, :, 0]
    b = ref_lab[:, :, 0]

    a = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((100, 100), np.float32) / 10000
    b = cv2.absdiff(b, cv2.filter2D(b, -1, kernel))
    a = cv2.absdiff(a, cv2.filter2D(a, -1, kernel))

    # Calculate diff
    diff_image = cv2.absdiff(a, b)

    return a, b, diff_image #cv2.cvtColor(diff_image, cv2.COLOR_BGR2GRAY)

def make_bb2(image, reference, mask):

    # Get grayscale
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

    # # Equalise images
    # img = cv2.equalizeHist(img)
    
    # hist1 = cv2.calcHist([img], [0], None, [256], [0, 256])
    # hist2 = cv2.calcHist([ref], [0], None, [256], [0, 256])

    # cdf1 = hist1.cumsum()
    # cdf2 = hist2.cumsum()

    # cdf1_normalized = cdf1 / cdf1.max()
    # cdf2_normalized = cdf2 / cdf2.max()

    # mapping = np.interp(cdf1_normalized, cdf2_normalized, np.arange(256))

    # equalized_image1 = mapping[img]
    # img = np.uint8(equalized_image1)

    # Add mask
    img *= mask
    ref *= mask

    # Take difference
    diff_image = cv2.absdiff(img, ref)

    # Extract important areas
    #   Get threshold
    obj_map = diff_image
    _, obj_map = cv2.threshold(obj_map, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #   Keep only important areas
    obj_image = diff_image * obj_map


    # Extract polution
    #   Get polution map
    polution_map = obj_image

    kernel = np.ones((100, 100), np.float32) / 10000
    polution_map = cv2.filter2D(polution_map, -1, kernel)

    _, polution_map = cv2.threshold(polution_map, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)



    # Get grayscale
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) * polution_map
    ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY) * polution_map

    # Equalise images
    img = cv2.equalizeHist(img)
    
    hist1 = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([ref], [0], None, [256], [0, 256])

    cdf1 = hist1.cumsum()
    cdf2 = hist2.cumsum()

    cdf1_normalized = cdf1 / cdf1.max()
    cdf2_normalized = cdf2 / cdf2.max()

    mapping = np.interp(cdf1_normalized, cdf2_normalized, np.arange(256))

    equalized_image1 = mapping[img]
    img = np.uint8(equalized_image1)

    # Add mask
    img *= mask
    ref *= mask

    # Take difference
    diff_image2 = cv2.absdiff(img, ref)

    int_img = diff_image2 * polution_map
    real_min = np.min(int_img[int_img > 0])
    int_img[int_img < 1] = real_min

    _, th2 = cv2.threshold(int_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return diff_image * polution_map, polution_map * 255, th2

    # Get polution
    polution = diff_image
    _, polution = cv2.threshold(polution, 110, np.max(polution), cv2.THRESH_BINARY)

    polution = cv2.absdiff(diff_image, polution)

    kernel = np.ones((100, 100), np.float32) / 10000
    polution = cv2.filter2D(polution, -1, kernel)

    #   Remove low value noise
    # _, pol_mask = cv2.threshold(polution, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # polution = polution * pol_mask

    polution = np.max(polution) - polution
    pol_mask = polution / np.max(polution)

    # Remove polution
    thr_img = (diff_image * pol_mask).astype(np.uint8)
    
    # Threshold image
    _, thr_img = cv2.threshold(thr_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morthology
    mortho_img = thr_img

    #   Remove granularity
    # kernel = np.ones((3, 3),np.uint8) / 9
    # mortho_img = cv2.erode(mortho_img, kernel, iterations=10)
    # mortho_img = cv2.dilate(mortho_img, kernel, iterations=10)


    #   Remove small features
    # kernel = np.ones((10, 10),np.uint8) / 100
    # mortho_img = cv2.erode(mortho_img, kernel, iterations=1)
    # mortho_img = cv2.dilate(mortho_img, kernel, iterations=2)

    return diff_image, polution, thr_img

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(mortho_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



    # Make bounding boxes
    bbs = [[*cv2.boundingRect(contour)] for contour in contours]

    # Get area and of bounding boxes
    areas = [bb[2] * bb[3] for bb in bbs]

    # Numpy arrays for ease
    bbs = np.array(bbs)
    areas = np.array(areas)
    
    # Check if bbs
    if len(bbs.shape) < 2:
        return diff_image, thr_img, image

    # Switch to x, y, xp, yp
    bbs[:, 2] = bbs[:, 0] + bbs[:, 2]
    bbs[:, 3] = bbs[:, 1] + bbs[:, 3]

    # Filter by size
    area_th = 5000
    mask_area = areas > area_th

    bbs = bbs[mask_area, :]
    areas = areas[mask_area]

    # Combine by overlap
    def combine_bbs(sim_bbs):

        x = sim_bbs[:, 0].min()
        y = sim_bbs[:, 1].min()
        xp = sim_bbs[:, 2].max()
        yp = sim_bbs[:, 3].max()

        return [x, y, xp, yp]

    def overlap(a, b):

        intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))

        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])

        return intersection / min(area_a, area_b)


    def get_all_overlap(bb, others):

        combine = []
        exclude = []
        for b in others:
            
            if overlap(bb, b) > 0.1:
                combine.append(b)
            else:
                exclude.append(b)
        
        return combine, exclude
    
    bbs = bbs.tolist()
    clean_run = False
    while not clean_run:
        clean_run = True
        new_bbs = []
        while len(bbs) > 0:

            bb = bbs.pop(0)
            combine, bbs = get_all_overlap(bb, bbs)

            if len(combine) > 0:
                clean_run = False

            combine.append(bb)
            new = combine_bbs(np.array(combine))
            new_bbs.append(new)
        
        bbs = new_bbs



    # Iterate through the detected contours and draw bounding boxes
    for bb in bbs:
        x, y, xp, yp = bb
        cv2.rectangle(image, (x, y), (xp, yp), (0, 255, 0), 5)

    return diff_image, mortho_img, image


mkbb = make_bb(load_other_images(folders[0])[5], load_reference_image(folders[0]), load_mask(folders[0])[:, :, 0])



cv2.imshow('make bb 1', cv2.resize(mkbb[0], (780, 540),
               interpolation = cv2.INTER_LINEAR))
cv2.imshow('make bb 2', cv2.resize(mkbb[1], (780, 540),
               interpolation = cv2.INTER_LINEAR))
cv2.imshow('make bb 3', cv2.resize(mkbb[2], (780, 540),
               interpolation = cv2.INTER_LINEAR))
cv2.waitKey(0)