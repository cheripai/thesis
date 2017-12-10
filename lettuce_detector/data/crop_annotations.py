import json
import numpy as np
import os
import sys
from PIL import Image
from scipy.misc import imsave

output_dir = "cropped"

if __name__ == "__main__":
    img_path = sys.argv[1]
    annotations_path = sys.argv[2]
    row_order_path = None

    if len(sys.argv) > 3:
        row_order_path = sys.argv[3]
        with open(row_order_path) as f:
            row_order = [row.strip() for row in f.readlines()]

    img = np.asarray(Image.open(img_path), dtype=np.uint8)
    annotations = json.load(open(annotations_path))[0]["annotations"]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, annotation in enumerate(annotations):
        x, y, height, width = int(annotation["x"]), int(annotation["y"]), int(annotation["height"]), int(annotation["width"])
        if row_order_path is not None:
            imsave(os.path.join(output_dir, row_order[i]+".png"), img[y:y+height, x:x+width])
        else:
            imsave(os.path.join(output_dir, str(i)+".png"), img[y:y+height, x:x+width])
