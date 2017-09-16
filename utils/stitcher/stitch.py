#
# Takes the directory that contains subdirectories of the tiles
#

import numpy as np
import os
import re
import scipy.misc
import sys
from PIL import Image
from scipy import ndimage


def get_png_files(directory):
    png_files = []
    for root, dirs, files in os.walk(directory):
        dirs.sort()
        for f in sorted(files):
            if f.endswith(".png"):
                png_files.append(os.path.join(root, f))
    return png_files

def get_size(directory):
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.endswith(".png"):
                img = Image.open(os.path.join(root, f))
                return img.size[0]

def get_rows_cols(directory):
    for root, dirs, files in os.walk(directory):
        cols = len(dirs)
        # Divide by 2 because directory contains a png and kml file for each tile
        rows = len([f for f in os.listdir(os.path.join(root, dirs[0]))]) // 2
        return rows, cols
    

if __name__ == "__main__":
    directory = sys.argv[1]
    size = get_size(directory)
    rows, cols = get_rows_cols(directory)
    print(size, rows, cols)

    png_files = get_png_files(directory)
    img = np.zeros((size * rows, size * cols, 3))

    k = 0
    for i in range(cols):
        for j in reversed(range(rows)):
            img[j*size:(j+1)*size, i*size:(i+1)*size, :] = ndimage.imread(png_files[k], mode="RGB")
            k += 1

    scipy.misc.imsave(sys.argv[2], img)
