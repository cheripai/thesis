""" Converts grayscale UAV image to NDVI values based on min and max NDVI.
"""

import cv2
import numpy as np
import sys


def gray2ndvi(img, min_ndvi, max_ndvi):
    ndvi = np.zeros(gray.shape)
    step = np.zeros(gray.shape)

    step[:] = (max_ndvi - min_ndvi) / 255
    ndvi = np.multiply(gray, step)

    return ndvi

if __name__ == "__main__":
    img_path = sys.argv[1]
    min_ndvi = float(sys.argv[2])
    max_ndvi = float(sys.argv[3])

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ndvi = gray2ndvi(gray, min_ndvi, max_ndvi)
