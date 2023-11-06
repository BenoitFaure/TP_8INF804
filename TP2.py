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
from sklearn.cluster import KMeans, AffinityPropagation, BisectingKMeans
from skimage import graph
from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering
from sklearn.decomposition import PCA

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

    # Section img Augmentation
    if True:

        # Augment contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_clahe = clahe.apply(gray)

        gray = gray_clahe
    
    # img preprocess
    image_prepross = np.zeros_like(img, np.uint8)
    if True:

        for i in range(3):
            image_prepross[:, :, i] = cv2.equalizeHist(img[:, :, i])

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
        # cleaned_image = cv2.bitwise_and(img, img, mask=clean_1)

        bkg = thresh_hold_1

    # Color spaces
    if True:
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_hsv_gray = img_hsv.mean(axis=2).astype(np.uint8)
        
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        img_lab_gray = img_lab.mean(axis=2).astype(np.uint8)

        img_xyz = cv2.cvtColor(img, cv2.COLOR_BGR2XYZ)
        img_xyz_gray = img_xyz.mean(axis=2).astype(np.uint8)

        # return img_xyz, img_xyz_gray

        # return gray, np.dstack([img_hsv_gray, img_lab_gray]).mean(axis=2).astype(np.uint8) * bkg

        # return img_lab, img_lab_gray * bkg

        # return gray * bkg, img_hsv_gray * bkg


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
        watershed = cv2.watershed(img, markers)
        filled = watershed#.astype(np.uint8) * 255

    # Watershed method
    if False:

        gray_source = img_lab_gray
        img_source = img_lab

        # Extract confused objects
        # ret, thresh_hold_1 = cv2.threshold(gray_source, 50, 255, cv2.THRESH_BINARY)
        ret, thresh_hold_1 = cv2.threshold(gray_source * bkg, 1, 255, cv2.THRESH_BINARY)

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
        
        ret, sure_fg = cv2.threshold(dist_transform,0.4*dist_transform.max(),255,0)
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

        ws_seg = cv2.watershed(img_source, markers)

        segmentation = ws_seg

    # Watershed method 1.1
    if False:

        gray_source = img_lab_gray
        img_source = img_lab

        # Extract confused objects
        # ret, thresh_hold_1 = cv2.threshold(gray_source, 50, 255, cv2.THRESH_BINARY)
        ret, thresh_hold_1 = cv2.threshold(gray_source * bkg, 1, 255, cv2.THRESH_BINARY)

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

        # Dilate dist_transform
        # kernel = np.ones((3, 3), np.uint8) / 9
        # kernel = cv2.getGaussianKernel(30, 1)
        # kernel = np.outer(kernel, kernel)
        # kernel /= kernel.sum()
        # # dist_transform_gauss = cv2.dilate(dist_transform, kernel, iterations=3)
        # dist_transform_gauss = cv2.filter2D(dist_transform, -1, kernel)
        # dist_transform_gauss = cv2.GaussianBlur(dist_transform, (31, 31), 0)

        kernel = np.array([[1, 1, 1], [1, 5, 1], [1, 1, 1]])
        kernel = kernel / (kernel.sum() * 0.9)
        dist_transform_gauss = cv2.filter2D(dist_transform, -1, kernel)

        dist_transform = dist_transform_gauss
        
        ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
        # Unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)

        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers+1
        markers[unknown==255] = 0


        # ws_img = image_prepross
        # ws_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # ws_img = cv2.cvtColor(image_prepross, cv2.COLOR_BGR2LAB)

        # Blur img
        ws_img = cv2.GaussianBlur(img_source, (101, 101), 0)

        ws_seg = cv2.watershed(ws_img, markers)

        segmentation = ws_seg

        return ws_img, segmentation / segmentation.max()

    # Blur for feature extraction
    if True:

        # Clean img
        cleaned_image = cv2.bitwise_and(img, img, mask=bkg)

        # Blur img
        blured_image = cv2.GaussianBlur(cleaned_image, (101, 101), 0)

        # Convert to LAB
        blured_image_lab = cv2.cvtColor(blured_image, cv2.COLOR_BGR2LAB)
        blured_image_hsv = cv2.cvtColor(blured_image, cv2.COLOR_BGR2HSV)

        # Cluster pixels by AB channels
        # pixel_features = blured_image_lab[:, :, 1:].reshape(-1, 2)
        msk = bkg != 0
        # pixel_features = blured_image_lab[msk, 1:]
        pixel_features = np.concatenate((blured_image_lab, blured_image_hsv), axis=2)[msk, :]
        labels = KMeans(n_clusters=20, n_init='auto').fit_predict(pixel_features)
        
        # Reshape to 2D array
        # labels = labels.reshape(blured_image_lab.shape[:2])
        n_img = np.zeros_like(blured_image_lab[:, :, 1])
        c = 0
        for i in range(n_img.shape[0]):
            for j in range(n_img.shape[1]):
                if msk[i, j]:
                    n_img[i, j] = labels[c]
                    c += 1

        # return blured_image_lab, n_img / n_img.max()

    # Slic + Feature + Clustering
    if True:
        # Skimage SLIC
        if True:

            # source_img = image_prepross
            source_img = img_xyz
            # source_img = blured_image

            # Slic Segmentation
            slic_seg = slic(source_img, n_segments=300, compactness=20, sigma=1)

            # Apply background
            segmentation = slic_seg * bkg

        
        # Section to try and explore line directions
        if False:

            gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

            gradient_direction = np.arctan2(gradient_y, gradient_x)
            norm_gradiant_direction = gradient_direction + np.pi
            norm_gradiant_direction = norm_gradiant_direction / (2 * np.pi)
            gradient_direction = norm_gradiant_direction * 255

            # return gradient_direction, norm_gradiant_direction

            grad_disp = np.zeros_like(gradient_direction, np.uint8)

            for j in range(1, segmentation.max() + 1):

                superpixel_mask = segmentation == j
                if superpixel_mask.sum() == 0:
                    continue

                grad_disp[superpixel_mask] = gradient_direction[superpixel_mask].mean()


            return gradient_direction, grad_disp


        # Clean segmentation - doesnt work well, should maybe do it after combining superpixels
        if True:

            # Remove dark objects that havent been removed by bkg extraction (like reflections)
            for i in range(1, segmentation.max() + 1):
                super_pix = cv2.bitwise_and(img, img, mask=(segmentation == i).astype(np.uint8))
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

            # Get all img channels
            image_hsv = img_hsv
            image_lab = img_lab
            B = img[:, :, 0]
            G = img[:, :, 1]
            R = img[:, :, 2]

            # channels = [image_hsv]
            # channels = [image_lab, image_hsv]
            # channels = [image_lab, blured_image]
            # channels = [img_xyz]
            channels = [image_lab, img_xyz]
            # channels = [image_lab, image_prepross]

            print("Extracted Channels")

            num_iters = 1
            cluster_models  = [
                AffinityPropagation(random_state=42),
                BisectingKMeans(n_clusters=20, random_state=42)
            ]
            for clustering_iter in range(num_iters):

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
                    # deg_of_lum = np.mean(0.299*R[superpixel_mask] + 0.587*G[superpixel_mask] + 0.114*B[superpixel_mask], axis=0)
                    # features.append(deg_of_lum)
                    # Add the blured img features
                    # features.append(np.argmax(np.bincount(n_img[superpixel_mask])))
                    # Add superpixel location
                    # features.extend(np.mean(np.argwhere(superpixel_mask), axis=0))

                    # Store superpixel and feature
                    super_pixels.append((j, superpixel_mask))
                    super_pixels_features.append(features)

                    # Show color histogram for superpixel
                    # plt.hist(img[superpixel_mask, 1].ravel(), 256, [0, 256])
                    # plt.show()
                    # exit()
                
                super_pixels_features = np.array(super_pixels_features)

                # PCA visualization
                # red_dim_feat_2d = PCA(n_components=2).fit_transform(super_pixels_features)
                # # Plot clusters
                # plt.scatter(red_dim_feat_2d[:, 0], red_dim_feat_2d[:, 1])
                # plt.show()
                # exit()

                print(f"Extracted Features {clustering_iter}/{num_iters}")

                base_segmented_image = mark_boundaries(img, segmentation)
                
                # Do clustering on features
                # cluster_model = KMeans(n_clusters=15, n_init='auto')
                cluster_model = AffinityPropagation(random_state=42)
                # cluster_model = BisectingKMeans(n_clusters=10, random_state=42)
                if num_iters > 1:
                    cluster_model = cluster_models[num_iters % len(cluster_models)]
                super_pixels_cluster = cluster_model.fit_predict(super_pixels_features) + 1
                
                # Assign ids to superpixels
                for i in range(len(super_pixels)):
                    segmentation[segmentation == super_pixels[i][0]] = super_pixels_cluster[i]

                print(f"Clustered {clustering_iter}/{num_iters}")

    # Cuts
    if False:
                
        labels1 = slic(img, compactness=30, n_segments=400, start_label=1)
        out1 = label2rgb(labels1, img, kind='avg', bg_label=0)

        g = graph.rag_mean_color(img, labels1, mode='similarity')
        labels2 = graph.cut_normalized(labels1, g)
        out2 = label2rgb(labels2, img, kind='avg', bg_label=0)

    # Spectral clustering - Hella slow havent been able to run it
    if False:
        gray_blured_img = cv2.cvtColor(blured_image, cv2.COLOR_BGR2GRAY)
        # resize to 0.2 of original size
        gray_blured_img = cv2.resize(gray_blured_img, (0, 0), fx=0.2, fy=0.2)
        img_graph = image.img_to_graph(gray_blured_img)

        beta = 10
        eps = 1e-6
        img_graph.data = np.exp(-beta * img_graph.data / img_graph.data.std()) + eps

        n_regions = 16
        n_regions_plus = 3

        labels = spectral_clustering(
            img_graph,
            n_clusters=(n_regions + n_regions_plus),
            eigen_tol=1e-7,
            assign_labels="kmeans",
            random_state=42,
        )

        labels = labels.reshape(gray_blured_img.shape)

        print(labels)
        exit()

        return blured_image, 

    # return out1, out2

    # segmented_image = label2rgb(segmentation, img, kind='avg')
    segmented_image = mark_boundaries(img, segmentation)

    return base_segmented_image, segmented_image
    return image_prepross, segmented_image
    return image_prepross, segmentation / segmentation.max()
    return img, image_cont

    # return dist_transform, canny_edges_dst
    # return dist_transform.astype(np.uint8) , sure_fg
    # return unknown, (segmentation + 1) / (segmentation.max() + 1) * 255

def run():
    images = load_images()

    img = images[0]

    img1, img2 = segment_image(img[0])
    cv2.imshow('Clean', img1)
    cv2.imshow('Segmented', img2)

    cv2.imshow(img[1], img[0])

    def test():
        im_lab = cv2.cvtColor(img[0], cv2.COLOR_BGR2LAB)

        # Show 3rd channel as hist
        plt.hist(im_lab[:, :, 2].ravel(), 256, [0, 256])
        plt.show()

        ret, th_1 = cv2.threshold(im_lab[:, :, 2], 140, 255, cv2.THRESH_BINARY)
        ret, th_2 = cv2.threshold(im_lab[:, :, 2], 137, 255, cv2.THRESH_BINARY)
        ret, th_3 = cv2.threshold(im_lab[:, :, 2], 130, 255, cv2.THRESH_BINARY)

        return th_1, th_2, th_3

        return im_lab[:, :, 0], im_lab[:, :, 1], im_lab[:, :, 2]
        return img[0][:, :, 0], img[0][:, :, 1], img[0][:, :, 2]

    # A, B, C = test()

    # cv2.imshow('A', A)
    # cv2.imshow('B', B)
    # cv2.imshow('C', C)
    # cv2.imshow('concat', (A/3 + B/3 + C/3).astype(np.uint8))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run()