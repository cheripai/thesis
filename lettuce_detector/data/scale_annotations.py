import json
import numpy as np
import os
import sys
from PIL import Image


if __name__ == "__main__":
    src_annotation_path = sys.argv[1]
    tgt_img_dir = sys.argv[2]
    out_path = sys.argv[3]

    annotations = json.loads(open(src_annotation_path).read())
    src_widths = []
    tgt_widths = []
    for annotation in annotations:
        src_img = np.array(Image.open(annotation["filename"]))
        src_widths.append(src_img.shape[1])

    for img_path in os.listdir(tgt_img_dir):
        if img_path.endswith("png"):
            tgt_img = np.array(Image.open(os.path.join(tgt_img_dir, img_path)))
            tgt_widths.append(tgt_img.shape[1])

    avg_src_width = np.average(src_widths)
    avg_tgt_width = np.average(tgt_widths)
    
    scaling_factor = avg_tgt_width / avg_src_width

    for annotation in annotations:
        for box in annotation["annotations"]:
            box["height"] *= scaling_factor
            box["width"] *= scaling_factor
            box["x"] *= scaling_factor
            box["y"] *= scaling_factor
        annotation["filename"] = os.path.join(tgt_img_dir, annotation["filename"].split("/")[-1])

    with open(out_path, "w") as f:
        f.write(json.dumps(annotations))
