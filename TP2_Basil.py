import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import itertools
def displayImage(img, title):
    cv.imshow(title, img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def plotImages(imgs, title):
    img_concats = np.concatenate(imgs, axis=1)
    cv.imshow(title, img_concats)
    cv.waitKey(0)
    cv.destroyAllWindows()

def saturate(img, percentage = 1.5):
    hsv_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Define a scaling factor for saturation (1.0 for no change, <1.0 for desaturation, >1.0 for saturation increase)
    saturation_scale = percentage

    # Scale the saturation channel
    hsv_image[..., 1] = np.clip(hsv_image[..., 1] * saturation_scale, 0, 255).astype(np.uint8)

    # Convert the modified HSV image back to BGR color space
    saturated_image = cv.cvtColor(hsv_image, cv.COLOR_HSV2BGR)
    return saturated_image

thresh = [70] #30-90
opening_dilate_iter_count = [1] #1-4
sure_fg_thresh = [0.4] #0.3-0.8

all_triplets = list(itertools.product(thresh, opening_dilate_iter_count, sure_fg_thresh))
for triplet in all_triplets:
    img = cv.imread('images/TP2/Echantillion1Mod2_301.png')
    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize = (16, 16))
    gray_image_clahe = clahe.apply(gray_image)

    ret, thresh = cv.threshold(gray_image_clahe, triplet[0], 255, cv.THRESH_BINARY)


    # noise removal
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    #kernel = np.ones((3, 3),np.uint8)

    opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
    closing = cv.morphologyEx(opening,cv.MORPH_CLOSE,kernel, iterations = 7)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    opening = closing

    kernel = np.ones((3, 3),np.uint8)
    opening = cv.erode(opening, kernel, iterations = 9)
    # sure background area
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    sure_bg = cv.dilate(opening,kernel,iterations=30)
    #displayImage(sure_bg, "Sure bg")

    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
    dist_output = cv.normalize(dist_transform, None, 0, 1.0, cv.NORM_MINMAX) 
    #displayImage(dist_output, "dist output")
    ret, sure_fg = cv.threshold(dist_transform,0.2*dist_transform.max(),255,0) #0.5 - 0.7
    #displayImage(sure_fg, "sure_fg")


    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg,sure_fg)
    #displayImage(unknown, "unknown region")
    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    markers = cv.watershed(img,markers)
    img[markers == -1] = [255,0,0]
    displayImage(img, f"result {triplet[0]} {triplet[1]} {triplet[2]}")
    plotImages((thresh, sure_bg, closing), f"Thresh, sure_bg, CLosing {triplet[0]} {triplet[1]} {triplet[2]}")
    plotImages((opening, dist_output, sure_fg, unknown), f"opening, dist_output, sure_fg, unknown {triplet[0]} {triplet[1]} {triplet[2]}")
    