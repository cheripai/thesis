# 
# Splits input image into tiles and saves annotations in csv
# Usage: python split_map.py <path/to/map.png> <path/to/annotations.json> <path/to/output/dir>
#

import json
import numpy as np
import os
import sys
from PIL import Image
from scipy.misc import imsave

column_names = ["filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"]
class_name = "lettuce"
tile_size = 1700

def reshape_image(img, tile_size, max_cut=50):
    height = img.shape[0]
    width = img.shape[1]
    h_remainder = height % tile_size
    w_remainder = width % tile_size
    if h_remainder < max_cut:
        height -= h_remainder
    else:
        height += tile_size - h_remainder
    if w_remainder < max_cut:
        width -= w_remainder
    else:
        width += tile_size - w_remainder
    img = img[:height, :width, :].copy()
    img.resize((height, width, img.shape[2]))
    return img


if __name__ == "__main__":
    img_path = sys.argv[1]
    annotations_path = sys.argv[2]
    output_dir = sys.argv[3]

    img = np.asarray(Image.open(img_path)).copy()
    img = reshape_image(img, tile_size)

    tile_annotations = [[[] for j in range(img.shape[1]//tile_size)] for i in range(img.shape[0]//tile_size)]
    rows = len(tile_annotations)
    cols = len(tile_annotations[0])

    annotations = json.loads(open(annotations_path).read())[0]["annotations"]
    for annotation in annotations:
        tile_annotations[int(annotation["y"] // tile_size)][int(annotation["x"] // tile_size)].append(annotation)

    with open(os.path.join(output_dir, "annotations.csv"), "w+") as f:
        f.write(",".join(column_names) + "\n")

        for row in range(rows):
            for col in range(cols):
                tile = img[row*tile_size:(row+1)*tile_size, col*tile_size:(col+1)*tile_size, :]
                tile_name = "{}_{}.jpg".format(row, col)
                imsave(os.path.join(output_dir, tile_name), tile)

                for annotation in tile_annotations[row][col]:
                    xmin = int(annotation["x"] - col * tile_size)
                    ymin = int(annotation["y"] - row * tile_size)
                    xmax = int(xmin + annotation["width"])
                    ymax = int(ymin + annotation["height"])
                    line = "{},{},{},{},{},{},{},{}".format(tile_name, tile_size, tile_size, class_name, xmin, ymin, xmax, ymax)
                    f.write(line + "\n")
