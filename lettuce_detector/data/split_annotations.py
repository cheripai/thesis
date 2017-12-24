import json
import os
import sys


def strip_filename(fname):
    return fname.split("/")[-1].split(".")[0].strip()


if __name__ == "__main__":
    annotation_fname = sys.argv[1]
    target_dir = sys.argv[2]
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    annotation_groups = json.load(open(annotation_fname))
    for group in annotation_groups:
        for annotation in group["annotations"]:
            annotation["class"] = "lettuce"
            annotation["height"] = round(annotation["height"])
            annotation["width"] = round(annotation["width"])
            annotation["x"] = round(annotation["x"])
            annotation["y"] = round(annotation["y"])
        target_fname = os.path.join(target_dir, strip_filename(group["filename"]) + ".json")
        json.dump(group["annotations"], open(target_fname, "w+"))
