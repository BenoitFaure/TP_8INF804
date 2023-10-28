from typing import List

import os

import cv2
import numpy as np

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

def run():
    images = load_images()

    image = images[0]
    cv2.imshow(image[1], image[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run()