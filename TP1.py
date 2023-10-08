from typing import List

import os
import re
import sys

import cv2
import numpy as np

# ------- IMAGE PROCESS -------
images_folder = '.\images'
folders = ['Chambre'    , 'Cuisine', 'Salon']

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
    filtered_files = [file for file in all_files if (file.lower().endswith('.jpg') and not file.lower().endswith('_bb.jpg')) and file != 'Reference.JPG' and file != 'Mask.JPG']

    full_path += '\\'
    images = [(cv2.imread(full_path + file), file) for file in filtered_files]

    return images

# ------- DEFINE TECHNIQUE -------

def extr_bb(img_c, image):

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(img_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Make bounding boxes
    bbs = [[*cv2.boundingRect(contour)] for contour in contours]

    # Get area and of bounding boxes
    areas = [bb[2] * bb[3] for bb in bbs]

    # Numpy arrays for ease
    bbs = np.array(bbs)
    areas = np.array(areas)
    
    # Check if bbs
    if len(bbs.shape) < 2:
        return image

    # Switch to x, y, xp, yp
    bbs[:, 2] = bbs[:, 0] + bbs[:, 2]
    bbs[:, 3] = bbs[:, 1] + bbs[:, 3]

    # Filter by size
    area_th = 7000
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
    
    return image, bbs

def make_bb(image, reference, mask):

    # Image to grayscale
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

    # Find contours
    kernel = np.ones((100, 100), np.float32) / 10000
    img_c = cv2.absdiff(img, cv2.filter2D(img, -1, kernel))
    ref_c = cv2.absdiff(ref, cv2.filter2D(ref, -1, kernel))

    # Calculate diff
    diff_c = cv2.absdiff(img_c, ref_c) * mask

    # Threshold
    thr_c = diff_c
    _, thr_c = cv2.threshold(thr_c, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morthology
    mortho_c = thr_c
    kernel = np.ones((4, 4),np.uint8) / 16
    mortho_c = cv2.erode(mortho_c, kernel, iterations=2)

    kernel = np.ones((30, 30),np.uint8) / 900
    mortho_c = cv2.dilate(mortho_c, kernel, iterations=2)

    return extr_bb(mortho_c, image)


# ------- BATCH PROCESS -------

for folder in folders:
    print(f'Starting folder {folder}')

    # Load images
    ref_img = load_reference_image(folder)
    mask = load_mask(folder)[:, :, 0]
    other_img = load_other_images(folder)

    # Calculate floor surface
    floor_surface = np.sum(mask > 0)

    # Progress bar
    n_img = len(other_img)
    c = 0

    for img, img_name in other_img:
        # Get bounding boxes for image
        img_bb, bbs = make_bb(img, ref_img, mask)

        # Calculate occupied area
        occupied_area = 0
        for bb in bbs:
            occupied_area += (bb[2] - bb[0]) * (bb[3] - bb[1])

        # Add info to image
        cv2.putText(img_bb, f"{len(bbs)} objects | Clutter {(occupied_area / floor_surface * 100):.2f}%", 
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 5, cv2.LINE_AA)

        # Save image
        cv2.imwrite(f"{images_folder}\{folder}\{re.sub('.JPG', '_bb.JPG', img_name)}", img_bb)

        # Progress bar
        c += 1
        msg = f"[{'-' * c}{' ' * (n_img - c)}] {c}/{n_img}"
        sys.stdout.write(msg)
        sys.stdout.flush()
        sys.stdout.write("\b" * len(msg))