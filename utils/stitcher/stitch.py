import numpy as np
import os
import re
import scipy.misc
import sys
from scipy import ndimage


def get_png_files(directory):
    png_files = []
    for root, dirs, files in os.walk(directory):
        dirs.sort()
        for f in sorted(files):
            if f.endswith(".png"):
                png_files.append(os.path.join(root, f))
    return png_files


if __name__ == "__main__":
    cols = 18
    rows = 20
    size = 256

    png_files = get_png_files(sys.argv[1])
    img = np.zeros((size * rows, size * cols, 3))

    k = 0
    for i in range(cols):
        for j in reversed(range(rows)):
            img[j*size:(j+1)*size, i*size:(i+1)*size, :] = ndimage.imread(png_files[k], mode="RGB")
            k += 1

    scipy.misc.imsave(sys.argv[2], img)
