import numpy as np
import os
import pandas as pd
import sys
from osgeo import gdal
from scipy.misc import imsave


def geotiff2arrays(fname):
    try:
        data = gdal.Open(fname)
    except:
        raise Exception("Invalid file!")
    values = data.GetRasterBand(1).ReadAsArray()
    try:
        transparency = data.GetRasterBand(2).ReadAsArray()
    except:
        transparency = np.zeros(values.shape)
        trasparency[:] = 255
    return values, transparency

if __name__ == "__main__":
    infile = sys.argv[1]
    values, transparency = geotiff2arrays(infile)

    bins = np.arange(-1.0, 1.0, 2 / 256)
    binner = lambda x: np.argmin(np.abs(bins-x))
    vbin = np.vectorize(binner)

    img_array = np.zeros((values.shape[0], values.shape[1], 4))
    img_array[:, :, :3] = np.expand_dims(vbin(values), axis=-1)
    img_array[:, :, 3] = transparency

    imsave(sys.argv[2], img_array)
