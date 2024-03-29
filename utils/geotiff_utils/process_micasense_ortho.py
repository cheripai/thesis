# Usage: python process_micasense_ortho.py <INPUT_RASTER>
import cv2
import numpy as np
import os
import sys
from osgeo import gdal
from rotate_and_crop import rotate, find_crop
from scipy.misc import imsave

"""
Blue: 475 nm
Green: 560 nm
Red: 668 nm
Red Edge: 717 nm
NIR: 840 nm
"""

ROTATION = 39.5
EPSILON = 1e-20
MIN = -1
MAX = 1

def NDVI(R, NIR):
    return (NIR - R) / (NIR + R + EPSILON)

def RedEdge(R, RE):
    return RE / (R + EPSILON)

def EVI(B, R, NIR):
    return 2.5 * (NIR - R) / (NIR + EPSILON) + 6 * R - 7.5 * B + 1 

def MCARI(G, R, RE):
    return (RE - R) - 0.23 * (RE - G) * RE / (R + EPSILON)

def GNDVI(G, NIR):
    return (NIR - G) / (NIR + G + EPSILON)

def geotiff2arrays(fname):
    try:
        data = gdal.Open(fname)
    except:
        raise Exception("Invalid file!")

    rasters = np.zeros((data.RasterYSize, data.RasterXSize, 5))
    rasters[:,:,0] = data.GetRasterBand(1).ReadAsArray()
    rasters[:,:,1] = data.GetRasterBand(2).ReadAsArray()
    rasters[:,:,2] = data.GetRasterBand(3).ReadAsArray()
    rasters[:,:,3] = data.GetRasterBand(4).ReadAsArray()
    rasters[:,:,4] = data.GetRasterBand(5).ReadAsArray()

    # Scale from 16-bit to 0-1
    rasters = rasters / 2**16
    return rasters


if __name__ == "__main__":
    infile = sys.argv[1]

    rasters = geotiff2arrays(infile)

    rasters = rotate(rasters, ROTATION)

    transparency = np.sum(rasters, axis=-1)
    transparency[np.where(transparency > 0)] = 255
    x, y, w, h = find_crop(transparency.astype(np.uint8))
    rasters = rasters[y:y+h, x:x+w]
    np.save("rasters.npy", rasters)

    B, G, R, RE, NIR = rasters[:,:,0], rasters[:,:,1], rasters[:,:,2], rasters[:,:,3], rasters[:,:,4]

    rgb = np.zeros((R.shape[0], R.shape[1], 3))
    rgb[:, :, 0] = R
    rgb[:, :, 1] = G
    rgb[:, :, 2] = B
    imsave("rgb.png", rgb)
