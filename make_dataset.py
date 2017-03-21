import bcolz
import json
import numpy as np
import os
import shutil
import sys
from keras.preprocessing import image


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


def get_ndvi(path):
    fname_ndvi = {}
    for d in os.listdir(path):
        p = os.path.join(path, d, "ndvi.txt")
        ndvis = [float(ndvi.strip()) for ndvi in open(p).readlines()]
        fnames = [os.path.join(d, fname) for fname in sorted(os.listdir(os.path.join(path, d))) if fname.endswith(VALID_EXT)]
        fname_ndvi = {**fname_ndvi, **dict(zip(fnames, ndvis))}

    cleaned_fname_ndvi = {}
    # NDVI of 0 or 1 means data error
    for fname, ndvi in fname_ndvi.items():
        if ndvi != 0.0 and ndvi != 1.0:
            cleaned_fname_ndvi[fname] = ndvi
    return cleaned_fname_ndvi


if sys.argv[1] == "bb":
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
    get_data = get_bbs
elif sys.argv[1] == "ndvi":
    DATA_PATH = "data/NDVI_Predictor"
    TRAIN_PATH = "data/train_ndvi"
    VALID_PATH = "data/valid_ndvi"
    TRAIN_DATA = "data/train_ndvi/train.dat"
    TRAIN_TARGET_DATA = "data/train_ndvi/train_target.dat"
    VALID_DATA = "data/valid_ndvi/valid.dat"
    VALID_TARGET_DATA = "data/valid_ndvi/valid_target.dat"
    SIZE = (4000, 3000)
    TARGET_SIZE = (224, 224)
    VALID_PROP = 0.15
    VALID_EXT = (".jpg")
    get_data = get_ndvi




if __name__ == "__main__":
    datas = get_data(DATA_PATH)
    train_datas = {}
    valid_datas = {}

    if not os.path.exists(TRAIN_PATH):
        os.makedirs(TRAIN_PATH)
    if not os.path.exists(VALID_PATH):
        os.makedirs(VALID_PATH)

    # Copy and rename image files to new location
    keys_datas = list(datas.items())
    np.random.shuffle(keys_datas)
    for i, (key, data) in zip(range(len(datas)), keys_datas):
        filename = str(i) + ".jpg"
        if i < int(VALID_PROP * len(datas)):
            valid_datas[filename] = data
            shutil.copyfile(os.path.join(DATA_PATH, key), os.path.join(VALID_PATH, filename))
        else:
            train_datas[filename] = data
            shutil.copyfile(os.path.join(DATA_PATH, key), os.path.join(TRAIN_PATH, filename))

    with open(os.path.join(TRAIN_PATH, "labels.json"), "w") as f:
        json.dump(train_datas, f)
    with open(os.path.join(VALID_PATH, "labels.json"), "w") as f:
        json.dump(valid_datas, f)

    # Convert images to np array
    train = np.zeros((len(train_datas), 3, TARGET_SIZE[0], TARGET_SIZE[1]))
    valid = np.zeros((len(valid_datas), 3, TARGET_SIZE[0], TARGET_SIZE[1]))
    train_target, valid_target = [], []
    for i, (key, data) in zip(range(len(train_datas)), train_datas.items()):
        img = image.load_img(
            os.path.join(TRAIN_PATH, key), target_size=TARGET_SIZE)
        train[i] = image.img_to_array(img)
        train_target.append(data)
    for i, (key, data) in zip(range(len(valid_datas)), valid_datas.items()):
        img = image.load_img(
            os.path.join(VALID_PATH, key), target_size=TARGET_SIZE)
        valid[i] = image.img_to_array(img)
        valid_target.append(data)

    train_target = np.array(train_target)
    valid_target = np.array(valid_target)

    # Write arrays to disk
    bcolz.carray(train, rootdir=TRAIN_DATA, mode="w").flush()
    bcolz.carray(train_target, rootdir=TRAIN_TARGET_DATA, mode="w").flush()
    bcolz.carray(valid, rootdir=VALID_DATA, mode="w").flush()
    bcolz.carray(valid_target, rootdir=VALID_TARGET_DATA, mode="w").flush()
