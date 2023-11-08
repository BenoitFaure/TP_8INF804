from typing import List

import os
import sys

import cv2
import numpy as np
from skimage.segmentation import slic, mark_boundaries
from scipy.stats import skew, kurtosis
from sklearn.cluster import KMeans

def load_images(folder: str = './images/TP2') -> List[np.array]:
    """ Loads all the images in the given folder into a list

    Parameters
    ----------
    folder : str
        The folder the images are taken from
    """
    full_path = folder
    all_files = os.listdir(full_path)
    filtered_files = [file for file in all_files if (file.lower().endswith('.jpg') or file.lower().endswith('.png')) and not file.lower().__contains__('_seg.')]

    full_path += '\\'
    images = [(cv2.imread(full_path + file), file) for file in filtered_files]

    return images

def segment_image(img: np.array) -> np.array:
    """ Segments the given img

    Parameters
    ----------
    img : np.array
        The img to segment
    """

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # img preprocess
    image_prepross = np.zeros_like(img, np.uint8)
    for i in range(3):
        image_prepross[:, :, i] = cv2.equalizeHist(img[:, :, i])
        
    # Extract background
    _, thresh_hold_1 = cv2.threshold(gray, 50, 1, cv2.THRESH_BINARY)
    bkg = thresh_hold_1

    # Color spaces
    img_xyz = cv2.cvtColor(img, cv2.COLOR_BGR2XYZ)

    # Blur for pixel wise feature extraction
    # Clean img
    cleaned_image = cv2.bitwise_and(img, img, mask=bkg)

    # Blur img
    blured_image = cv2.GaussianBlur(cleaned_image, (101, 101), 0)

    # Convert to LAB
    blured_image_lab = cv2.cvtColor(blured_image, cv2.COLOR_BGR2LAB)
    blured_image_hsv = cv2.cvtColor(blured_image, cv2.COLOR_BGR2HSV)

    # Cluster pixels
    msk = bkg != 0
    pixel_features = np.concatenate((blured_image_lab, blured_image_hsv), axis=2)[msk, :]
    labels = KMeans(random_state=42, n_clusters=20, n_init='auto').fit_predict(pixel_features)
    
    # Reshape to 2D array
    n_img = np.zeros_like(blured_image_lab[:, :, 1])
    c = 0
    for i in range(n_img.shape[0]):
        for j in range(n_img.shape[1]):
            if msk[i, j]:
                n_img[i, j] = labels[c]
                c += 1

    # Slic + Feature + Clustering
    # SLIC Segmentation
    source_img = img_xyz
    slic_seg = slic(source_img, n_segments=300, compactness=20, sigma=1)

    # Apply background
    segmentation = slic_seg * bkg

    # Clean segmentation - Remove small superpixels
    for i in range(1, segmentation.max() + 1):
        if np.count_nonzero(segmentation == i) < 500:
            segmentation[segmentation == i] = 0

    # Function to extract features from superpixel
    def extract_features(superpixel: np.array) -> np.array:

        mean = np.mean(superpixel, axis=(0, 1))
        std = np.std(superpixel, axis=(0, 1))
        skewness = skew(superpixel, axis=(0, 1))
        kurt = kurtosis(superpixel, axis=(0, 1))

        channel_features = np.array([mean, std, skewness, kurt]).flatten().tolist()
        channel_features = np.array([mean, std]).flatten().tolist()

        # Add top histogram values
        for i in range(1 if len(superpixel.shape) == 2 else superpixel.shape[2]):
            hist_channel = cv2.calcHist([superpixel], [i], None, [256], [0, 256])[1:]
            channel_features.append(np.argmax(hist_channel) + 1)

        channel_features = np.array(channel_features)
        channel_features = np.nan_to_num(channel_features, 0)

        return channel_features

    # Get all img channels
    B = img[:, :, 0]
    G = img[:, :, 1]
    R = img[:, :, 2]
    
    channels = [img_xyz, image_prepross]

    super_pixels = []
    super_pixels_features = []

    # Loop through superpixels and extract features
    for j in range(1, segmentation.max() + 1):

        superpixel_mask = segmentation == j
        if superpixel_mask.sum() == 0:
            continue

        # Extract features
        features = []
        for channel in channels:
            features.extend(extract_features(channel[superpixel_mask]))
        #   Extract Deg of luminance
        deg_of_lum = np.mean(0.299*R[superpixel_mask] + 0.587*G[superpixel_mask] + 0.114*B[superpixel_mask], axis=0)
        features.append(deg_of_lum)
        # Add the blured img features
        features.append(np.argmax(np.bincount(n_img[superpixel_mask])))

        # Extend mask
        kernel = np.ones((3, 3), np.uint8) / 9
        superpixel_mask_ext = cv2.dilate(superpixel_mask.astype(np.uint8), kernel, iterations=2)

        # Store superpixel and feature
        super_pixels.append((j, superpixel_mask, superpixel_mask_ext))
        super_pixels_features.append(features)
    
    # Make array
    super_pixels_features = np.array(super_pixels_features)
    # Normalize
    super_pixels_features = (super_pixels_features - super_pixels_features.min(axis=0)) / (super_pixels_features.max(axis=0) - super_pixels_features.min(axis=0))

    # Combine superpixels
    for curr_pix in range(len(super_pixels)):
        for other_pix in range(curr_pix + 1, len(super_pixels)):
            # Check if superpixels overlap
            if np.any(np.logical_and(super_pixels[curr_pix][2], super_pixels[other_pix][2])):
                
                # Combine based on feature distance
                #   Check distance between features
                if np.linalg.norm(super_pixels_features[curr_pix] - super_pixels_features[other_pix]) < 0.35:
                    # Set other pixel to same id as current pixel
                    segmentation[segmentation == super_pixels[other_pix][0]] = super_pixels[curr_pix][0]
                    super_pixels[other_pix] = (super_pixels[curr_pix][0], super_pixels[other_pix][1], super_pixels[other_pix][2])

    return segmentation

def run(folder: str = './images/TP2'):
    images = load_images(folder)

    for img in images:

        seg = segment_image(img[0])
        seg_img = mark_boundaries(img[0], seg)

        # Save image to file
        cv2.imwrite(folder + '\\' + img[1].split('.')[0] + '_seg.png', seg_img * 255)

        # Extract moy on each channel per superpixel
        channels = [img[0][:, :, 0], img[0][:, :, 1], img[0][:, :, 2]]
        #   Open csv
        with open(folder + '\\' + img[1].split('.')[0] + '_seg.csv', 'w') as f:
            #   Write header
            f.write('Superpixel,Mean B,Mean G,Mean R\n')
            #   Loop through superpixels
            c = 0
            for i in range(1, seg.max() + 1):
                #   Get superpixel mask
                superpixel_mask = seg == i
                if superpixel_mask.sum() == 0:
                    continue

                vals = f"{c},"

                # Extract values
                for channel in channels:
                    vals += f"{np.mean(channel[superpixel_mask])},"
                
                # Write to file
                f.write(vals[:-1] + '\n')
        
                c += 1

        # cv2.imshow('img', seg_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # exit()

        print(f"Image {img[1]} done")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        run(folder=sys.argv[1])
    else:
        run()