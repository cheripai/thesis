# Crop annotations from Sloth into sub-images.
# If row_order_path is provided, will crop into rows.
# Otherwise, will crop into individual plant images.
# Usage: python crop_annotations.py <annotation_path> <row_order_path (optional)>
import json
import numpy as np
import os
import sys
from PIL import Image
from scipy.misc import imsave

output_dir = "cropped"

if __name__ == "__main__":
    annotations_path = sys.argv[1]
    row_order_path = None

    if len(sys.argv) > 2:
        row_order_path = sys.argv[2]
        with open(row_order_path) as f:
            row_order = [row.strip() for row in f.readlines()]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    annotations = json.load(open(annotations_path))

    if row_order_path is not None:
        img = np.asarray(Image.open(annotations[0]["filename"]), dtype=np.uint8)

        for i, bbox in enumerate(annotations[0]["annotations"]):
            x, y, height, width = int(bbox["x"]), int(bbox["y"]), int(bbox["height"]), int(bbox["width"])
            imsave(os.path.join(output_dir, row_order[i]+".png"), img[y:y+height, x:x+width])
    else:
        for annotation in annotations:
            img = np.asarray(Image.open(annotation["filename"]), dtype=np.uint8)
            bboxes = sorted(annotation["annotations"], key=lambda k: k["x"], reverse=True)
            identifier, row = annotation["filename"].split("/")[-1].split(".")[0].split("Row")
            row = int(row) * 100
            for i, bbox in enumerate(bboxes):
                x, y, height, width = int(bbox["x"]), int(bbox["y"]), int(bbox["height"]), int(bbox["width"])
                imsave(os.path.join(output_dir, identifier+"_"+str(row+i+1)+".png"), img[y:y+height, x:x+width])
