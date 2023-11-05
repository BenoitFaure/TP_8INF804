from typing import List

import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries
from skimage.color import label2rgb
from skimage.feature import canny
from scipy import ndimage as ndi
from skimage.filters import sobel
from skimage import morphology
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

def segment_image(image: np.array) -> np.array:
    """ Segments the given image

    Parameters
    ----------
    image : np.array
        The image to segment
    """

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Section Image Augmentation
    if True:

        # Augment contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_clahe = clahe.apply(gray)

        gray = gray_clahe

    # Extract background
    if True:
            
            # Extract background
            ret, thresh_hold_1 = cv2.threshold(gray, 50, 1, cv2.THRESH_BINARY)
    
            # # Fill in regions
            # kernel = np.ones((3, 3), np.uint8) / 9
            # fill_1 = cv2.dilate(thresh_hold_1, kernel, iterations=1)
            # fill_1 = cv2.erode(fill_1, kernel, iterations=1)
    
            # # Remove small objects
            # clean_1 = cv2.erode(fill_1, kernel, iterations=3)
            # clean_1 = cv2.dilate(clean_1, kernel, iterations=3)
    
            # # Remove background
            # cleaned_image = cv2.bitwise_and(image, image, mask=clean_1)
    
            bkg = thresh_hold_1


    # Section edge detection
    # Custom edge detection
    if False:
        # Get contours - apply blur
        a, b = 100, 100
        kernel = np.ones((a, b), np.float32) / (a * b)
        blur = cv2.filter2D(gray, -1, kernel)

        gray_edges = cv2.absdiff(gray, blur)

        # _, thr_c = cv2.threshold(gray_edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, thr_c = cv2.threshold(gray_edges, 10, 255, cv2.THRESH_BINARY)
        thr_c = 255 - thr_c

        # Find background
        ret, thresh_back = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

        # Remove background
        cleaned_countours = cv2.bitwise_and(thr_c, thresh_back)
        # cleaned_countours = canny(gray_edges/255)

        edge_out = cleaned_countours

    # Canny edge detection
    if False:
        canny_edges = canny(gray/255)
        canny_edges = canny_edges.astype(np.uint8) * 255

        edge_out = canny_edges

    # Fill Holes
    if False:
        # Fill holes
        filled = ndi.binary_fill_holes(edge_out > 0)
        filled = filled.astype(np.uint8) * 255

    # Height map - Custom
    if False:
        # Get contours - apply blur
        a, b = 100, 100
        kernel = np.ones((a, b), np.float32) / (a * b)
        blur = cv2.filter2D(gray, -1, kernel)

        gray_edges = cv2.absdiff(gray, blur)

        # Find background
        ret, thresh_back = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

        # Remove background
        cleaned_edges = cv2.bitwise_and(gray_edges, thresh_back)

        h_map = cleaned_edges

        markers = np.zeros_like(h_map, np.uint8)
        markers[h_map > 100] = 2
        markers[h_map < 50] = 1

    # Watershed method to fill
    if False:
        watershed = cv2.watershed(image, markers)
        filled = watershed#.astype(np.uint8) * 255

    # Watershed method
    if False:

        # Extract confused objects
        ret, thresh_hold_1 = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

        # Fill in regions
        kernel = np.ones((3, 3), np.uint8) / 9
        fill_1 = cv2.dilate(thresh_hold_1, kernel, iterations=1)
        fill_1 = cv2.erode(fill_1, kernel, iterations=1)

        # Remove small objects
        clean_1 = cv2.erode(fill_1, kernel, iterations=3)
        clean_1 = cv2.dilate(clean_1, kernel, iterations=3)

        # Sure background area
        sure_bg = cv2.dilate(clean_1,kernel,iterations=3)
        ret, sure_bg = cv2.threshold(sure_bg, 200, 255, cv2.THRESH_BINARY)

        # Sure foreground area
        dist_transform = cv2.distanceTransform(clean_1, cv2.DIST_L2, 5)
        dist_transform = dist_transform.astype(np.uint8)
        # dist_transform = dist_transform * (255 / dist_transform.max())

        # Remove edges
        #   Dilate edges
        # kernel = np.ones((3, 3), np.uint8) / 9
        # edge_out_dil = cv2.dilate(edge_out, kernel, iterations=3)

        # dist_transform = cv2.subtract(dist_transform, edge_out_dil)

        # Dilate dist_transform
        # kernel = np.ones((3, 3), np.uint8) / 9
        kernel = cv2.getGaussianKernel(30, 1)
        kernel = np.outer(kernel, kernel)
        kernel /= kernel.sum()
        # dist_transform_gauss = cv2.dilate(dist_transform, kernel, iterations=3)
        dist_transform_gauss = cv2.filter2D(dist_transform, -1, kernel)

        dist_transform = dist_transform_gauss

        # canny_edges_dst = canny(dist_transform / np.max(dist_transform))
        # canny_edges_dst = canny_edges_dst.astype(np.uint8) * 255
        
        ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
        # Unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)

        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers+1
        # markers = np.zeros_like(sure_fg, np.int32)
        # markers[sure_fg > 0] = 2
        # markers[sure_bg < 1] = 1
        markers[unknown==255] = 0

        # show_marker = np.zeros((markers.shape[0], markers.shape[1], 3), np.uint8)
        # show_marker[:, :, 0] = (markers == 0) * 255
        # show_marker[:, :, 1] = (markers == 1) * 255
        # show_marker[:, :, 2] = (markers == 2) * 255

        segmentation = cv2.watershed(image, markers)

    # Image preprocess
    image_prepross = np.zeros_like(image, np.uint8)
    if True:

        for i in range(3):
            image_prepross[:, :, i] = cv2.equalizeHist(image[:, :, i])


    # Skimage SLIC
    if True:

        # Slic Segmentation
        slic_seg = slic(image_prepross, n_segments=250, compactness=20, sigma=1)

        # Apply background
        segmentation = slic_seg * bkg

    # Clean segmentation - doesnt work well, should maybe do it after combining superpixels
    if False:

        # Remove dark objects that havent been removed by bkg extraction (like reflections)
        for i in range(1, segmentation.max() + 1):
            super_pix = cv2.bitwise_and(image, image, mask=(segmentation == i).astype(np.uint8))
            if super_pix.mean() < 20/255:
                segmentation[segmentation == i] = 0


    # Function to extract features from superpixel
    def extract_features(superpixel: np.array) -> np.array:

        mean = np.mean(superpixel, axis=0)
        std = np.std(superpixel, axis=0)
        skewness = skew(superpixel, axis=0)
        kurt = kurtosis(superpixel, axis=0)

        channel_features = np.array([mean, std, skewness, kurt]).flatten()
        channel_features = np.nan_to_num(channel_features, 0)

        return channel_features


    # Combine superpixels
    if True:

        # Get all image channels
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        B = image[:, :, 0]
        G = image[:, :, 1]
        R = image[:, :, 2]

        channels = [image_hsv, image_lab]

        print("Extracted Channels")

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
                print(channel.shape)
                features.extend(extract_features(channel[superpixel_mask]))
            #   Extract Deg of luminance
            deg_of_lum = np.mean(0.299*R[superpixel_mask] + 0.587*G[superpixel_mask] + 0.114*B[superpixel_mask], axis=0)
            features.append(deg_of_lum)

            # Store superpixel and feature
            super_pixels.append((j, superpixel_mask))
            super_pixels_features.append(features)

            # Show color histogram for superpixel
            plt.hist(image[superpixel_mask, 1].ravel(), 256, [0, 256])
            plt.show()
            exit()
        
        super_pixels_features = np.array(super_pixels_features) + 1

        print("Extracted Features")
        
        # Do clustering on features
        super_pixels_cluster = KMeans(n_init='auto').fit_predict(super_pixels_features)

        print("Clustered")

        base_segmented_image = mark_boundaries(image, segmentation)

        # Assign ids to superpixels
        for i in range(len(super_pixels)):
            segmentation[segmentation == super_pixels[i][0]] = super_pixels_cluster[i]



    # segmented_image = label2rgb(segmentation, image, kind='avg')
    segmented_image = mark_boundaries(image, segmentation)

    return base_segmented_image, segmented_image
    return image_prepross, segmentation / segmentation.max()
    return image, image_cont

    # return dist_transform, canny_edges_dst
    # return dist_transform.astype(np.uint8) , sure_fg
    # return unknown, (segmentation + 1) / (segmentation.max() + 1) * 255

def run():
    images = load_images()

    image = images[0]
    cv2.imshow(image[1], image[0])

    img1, img2 = segment_image(image[0])
    cv2.imshow('Clean', img1)
    cv2.imshow('Segmented', img2)

    def test():
        im_lab = cv2.cvtColor(image[0], cv2.COLOR_BGR2LAB)

        # Show 3rd channel as hist
        plt.hist(im_lab[:, :, 2].ravel(), 256, [0, 256])
        plt.show()

        ret, th_1 = cv2.threshold(im_lab[:, :, 2], 140, 255, cv2.THRESH_BINARY)
        ret, th_2 = cv2.threshold(im_lab[:, :, 2], 137, 255, cv2.THRESH_BINARY)
        ret, th_3 = cv2.threshold(im_lab[:, :, 2], 130, 255, cv2.THRESH_BINARY)

        return th_1, th_2, th_3

        return im_lab[:, :, 0], im_lab[:, :, 1], im_lab[:, :, 2]
        return image[0][:, :, 0], image[0][:, :, 1], image[0][:, :, 2]

    # A, B, C = test()

    # cv2.imshow('A', A)
    # cv2.imshow('B', B)
    # cv2.imshow('C', C)
    # cv2.imshow('concat', (A/3 + B/3 + C/3).astype(np.uint8))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run()