from typing import List

import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.color import label2rgb
from skimage.feature import canny
from scipy import ndimage as ndi
from skimage.filters import sobel

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

    # Get contours - apply blur
    a, b = 100, 100
    kernel = np.ones((a, b), np.float32) / (a * b)
    blur = cv2.filter2D(gray, -1, kernel)

    gray_edges = cv2.absdiff(gray, blur)

    canny_edges = canny(gray/255)
    canny_edges = canny_edges.astype(np.uint8) * 255

    # _, thr_c = cv2.threshold(gray_edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, thr_c = cv2.threshold(gray_edges, 10, 255, cv2.THRESH_BINARY)
    thr_c = 255 - thr_c

    # Find background
    ret, thresh_back = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    # Remove background
    cleaned_countours = cv2.bitwise_and(thr_c, thresh_back)
    # cleaned_countours = canny(gray_edges/255)

    # Fill holes
    filled = ndi.binary_fill_holes(canny_edges)
    filled = filled.astype(np.uint8) * 255

    # Apply Threshold - Triangle
    # ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)

    # Get histogram
    # hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

    # Apply Threshold
    # ret, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    # Plot color histogram
    # plt.hist(image.ravel(), 256, [0, 256])
    # plt.show()

    # # Apply Gaussian Blur
    # blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # # Apply Canny Edge Detection
    # edges = cv2.Canny(blur, 100, 200)

    # # Apply Dilation
    # kernel = np.ones((5, 5), np.uint8)

    # dilation = cv2.dilate(edges, kernel, iterations=1)

    # # Apply Erosion
    # erosion = cv2.erode(dilation, kernel, iterations=1)

    # # Apply Closing
    # closing = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel)

    # # Apply Opening
    # opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

    # # Apply Threshold
    # ret, thresh = cv2.threshold(opening, 127, 255, cv2.THRESH_BINARY)

    # Convert to LAB
    # lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Improve contrast on light channel of image using histogram equalization
    # lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])

    # Clean up light channel
    # lab[:, :, 0] = lab[:, :, 0] * (thresh // 255)

    # Conver lab to rgb
    # img_lab_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Clean up image
    # n_image = image
    # n_image[thresh == 0] = [0, 0, 0]
    # image = n_image.astype(np.uint8)

    # Sclic segmentation
    # segments = slic(lab, n_segments=500, sigma=5)

    # Final segmented rgb image
    # img_rgb = label2rgb(segments, image, kind='avg')

    # Threshold again
    # ret, thresh_seg = cv2.threshold(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return canny_edges, filled

def run():
    images = load_images()

    image = images[0]
    cv2.imshow(image[1], image[0])

    img1, img2 = segment_image(image[0])
    cv2.imshow('Clean', img1)
    cv2.imshow('Segmented', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run()