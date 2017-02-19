import json
import numpy as np
import os
import shutil


# DATA_PATH should have subdirectories of images with an annotations file
DATA_PATH = "data/RGB"
TRAIN_PATH = "data/train_bbox"
VALID_PATH = "data/valid_bbox"
SIZE = (4000, 3000)
VALID_PROP = 0.15


def get_bbs(path):
    # Load bounding box json files
    bb_json = {}
    empty_bbox = {"height": 0., "width": 0., "x": 0., "y": 0.}
    for d in os.listdir(path):
        p = os.path.join(path, d, "annotations.json")
        j = json.load(open(p))
        for l in j:
            if "annotations" in l.keys() and len(l["annotations"]) > 0:
                bb_json[l["filename"]] = l["annotations"][0]
            else:
                bb_json[l["filename"]] = empty_bbox

    # Convert bounding box coordinates to resized_image
    bbs = {}
    for key, bb in bb_json.items():
        bbs[key] = convert_bb(bb, SIZE)
    return bbs
    

def convert_bb(bb, size):
    bb_params = ["height", "width", "x", "y"]
    bb = [bb[p] for p in bb_params]
    conv_x  = (224. / size[0])
    conv_y = (224. / size[1])
    bb[0] = bb[0] * conv_y
    bb[1] = bb[1] * conv_x
    bb[2] = max(bb[2]*conv_x, 0)
    bb[3] = max(bb[3]*conv_y, 0)
    return bb


if __name__ == "__main__":
    bbs = get_bbs(DATA_PATH)
    train_bbs = {}
    valid_bbs = {}

    # Copy and rename image files to new location
    keys_bbs = list(bbs.items())
    np.random.shuffle(keys_bbs)
    for i, (key, bb) in zip(range(len(bbs)), keys_bbs):
        filename = str(i) + ".jpg"
        if i < int(VALID_PROP * len(bbs)):
            valid_bbs[filename] = bb
            shutil.copyfile(os.path.join(DATA_PATH, key), os.path.join(VALID_PATH, filename))
        else:
            train_bbs[filename] = bb
            shutil.copyfile(os.path.join(DATA_PATH, key), os.path.join(TRAIN_PATH, filename))

    with open(os.path.join(TRAIN_PATH, "annotations.json"), "w") as f:
        json.dump(train_bbs, f)
    with open(os.path.join(VALID_PATH, "annotations.json"), "w") as f:
        json.dump(valid_bbs, f)
