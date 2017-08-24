""" Converts grayscale UAV image to NDVI values based on min and max NDVI.
"""

import cv2
import numpy as np
import sys


def threshold(arr, min_ndvi, max_ndvi, t=48):
    step = (max_ndvi - min_ndvi) / 255
    arr[arr <= t*step + min_ndvi] = np.nan
    return arr


def gray2ndvi(img, min_ndvi, max_ndvi):
    ndvi = np.zeros(img.shape)
    step = np.zeros(img.shape)

    step[:] = (max_ndvi - min_ndvi) / 255
    ndvi = np.multiply(img, step) + min_ndvi

    return ndvi

if __name__ == "__main__":
    img_path = sys.argv[1]
    min_ndvi = float(sys.argv[2])
    max_ndvi = float(sys.argv[3])

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ndvi = gray2ndvi(gray, min_ndvi, max_ndvi)
    ndvi = threshold(ndvi, min_ndvi, max_ndvi)

    print(np.nanmean(ndvi))
