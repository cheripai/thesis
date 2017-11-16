import numpy as np
import os
import pandas as pd
import sys
from osgeo import gdal
from scipy.misc import imsave


def strip_filename(fname):
    return os.path.splitext(os.path.basename(fname))[0]

def geotiff2arrays(fname):
    try:
        data = gdal.Open(fname)
    except:
        raise Exception("Invalid file!")
    values = data.GetRasterBand(1).ReadAsArray()
    transparency = data.GetRasterBand(2).ReadAsArray()
    return values, transparency

if __name__ == "__main__":
    infile = sys.argv[1]
    values, transparency = geotiff2arrays(infile)

    min_value = values.min()
    max_value = values.max()

    bins = np.zeros(256)
    for i in range(256):
        bins[i] = (max_value - min_value) / 256 * i + min_value

    img_array = np.zeros((values.shape[0], values.shape[1], 4))
    binner = lambda x: np.argmin(np.abs(bins-x))
    vbin = np.vectorize(binner)
    img_array[:, :, :3] = np.expand_dims(vbin(values), axis=-1)
    img_array[:, :, 3] = transparency

    basename = strip_filename(infile)
    imsave(basename + ".png", img_array)
    np.savetxt(basename + "_bins.txt", bins, fmt="%1.4f")
