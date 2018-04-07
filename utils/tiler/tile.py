# 
# Tiles multiple input images into tiles where all tiles are from the same coordinates.
# Requires all input images to be georeferenced and of the same dimensions.
#
# Usage: python tile.py <image_0> <image_1> ... <image_n>
#

import numpy as np
import os
import sys
from PIL import Image
from random import randint
from scipy.misc import imsave
from sklearn.feature_extraction.image import extract_patches_2d

patch_size = 56

if __name__ == "__main__":
    for i in range(len(sys.argv) - 1):
        dir_name = "tiles_{}".format(i)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        img = np.asarray(Image.open(sys.argv[i + 1]), dtype=np.uint8)
        h, w = img.shape[:2]

        for j in range((h // patch_size) - 1):
            for k in range((w // patch_size) - 1):
                imsave("{}/{}_{}.png".format(dir_name, j, k), img[patch_size*j:patch_size*(j+1), patch_size*k:patch_size*(k+1)])
