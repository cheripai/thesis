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

patch_size = 256

if __name__ == "__main__":
    seed = randint(0, 1000)
    for i in range(len(sys.argv) - 1):
        dir_name = "tiles_{}".format(i)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        img = np.asarray(Image.open(sys.argv[i + 1]), dtype=np.uint8)
        h, w = img.shape[:2]
        max_patches = int(1.5 * h * w / patch_size**2)
        patches = extract_patches_2d(img, patch_size=(patch_size, patch_size), max_patches=max_patches, random_state=seed)
        for j in range(patches.shape[0]):
            imsave("{}/{}.png".format(dir_name, j), patches[j])
