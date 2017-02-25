import bcolz
import json
import numpy as np
import os
import shutil
from keras.preprocessing import image


# DATA_PATH should have subdirectories of images with an annotations file
DATA_PATH = "data/RGB"
TRAIN_PATH = "data/train_bbox"
VALID_PATH = "data/valid_bbox"
TRAIN_DATA = "data/train_bbox/train.dat"
TRAIN_TARGET_DATA = "data/train_bbox/train_target.dat"
VALID_DATA = "data/valid_bbox/valid.dat"
VALID_TARGET_DATA = "data/valid_bbox/valid_target.dat"
SIZE = (4000, 3000)
TARGET_SIZE = (224, 224)
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

    # Convert images to np array
    train = np.zeros((len(train_bbs), 3, TARGET_SIZE[0], TARGET_SIZE[1]))
    valid = np.zeros((len(valid_bbs), 3, TARGET_SIZE[0], TARGET_SIZE[1]))
    train_target, valid_target = [], []
    for i, (key, bb) in zip(range(len(train_bbs)), train_bbs.items()):
        img = image.load_img(
            os.path.join(TRAIN_PATH, key), target_size=TARGET_SIZE)
        train[i] = image.img_to_array(img)
        train_target.append(bb)
    for i, (key, bb) in zip(range(len(valid_bbs)), valid_bbs.items()):
        img = image.load_img(
            os.path.join(VALID_PATH, key), target_size=TARGET_SIZE)
        valid[i] = image.img_to_array(img)
        valid_target.append(bb)

    train_target = np.array(train_target)
    valid_target = np.array(valid_target)

    # Write arrays to disk
    bcolz.carray(train, rootdir=TRAIN_DATA, mode="w").flush()
    bcolz.carray(train_target, rootdir=TRAIN_TARGET_DATA, mode="w").flush()
    bcolz.carray(valid, rootdir=VALID_DATA, mode="w").flush()
    bcolz.carray(valid_target, rootdir=VALID_TARGET_DATA, mode="w").flush()
