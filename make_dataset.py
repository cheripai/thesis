import bcolz
import json
import numpy as np
import os
import shutil
import sys
from utils.utils import get_config
from keras.preprocessing import image


def data_to_array(fname_target, path, target_size=(224, 224)):
    array = np.zeros((len(fname_target), target_size[0], target_size[1], 3))
    targets = []
    for i, (fname, target) in zip(range(len(fname_target)), fname_target.items()):
        img = image.load_img(os.path.join(path, fname), target_size=target_size)
        array[i] = image.img_to_array(img)
        targets.append(target)
    return array, np.array(targets)


def get_bbs(path, size=(4000, 3000)):
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
        bbs[key] = convert_bb(bb, size)
    return bbs


def convert_bb(bb, size):
    bb_params = ["height", "width", "x", "y"]
    bb = [bb[p] for p in bb_params]
    conv_x = (224. / size[0])
    conv_y = (224. / size[1])
    bb[0] = bb[0] * conv_y
    bb[1] = bb[1] * conv_x
    bb[2] = max(bb[2] * conv_x, 0)
    bb[3] = max(bb[3] * conv_y, 0)
    return bb


def get_ndvi(path, valid_exts=(".jpg")):
    fname_ndvi = {}
    for d in os.listdir(path):
        p = os.path.join(path, d, "ndvi.txt")
        ndvis = [float(ndvi.strip()) for ndvi in open(p).readlines()]
        fnames = [
            os.path.join(d, fname) for fname in sorted(os.listdir(os.path.join(path, d))) if fname.endswith(valid_exts)
        ]
        fname_ndvi = {**fname_ndvi, **dict(zip(fnames, ndvis))}

    cleaned_fname_ndvi = {}
    # NDVI of 0 or 1 means data error
    for fname, ndvi in fname_ndvi.items():
        if ndvi != 0.0 and ndvi != 1.0:
            cleaned_fname_ndvi[fname] = ndvi
    return cleaned_fname_ndvi


if __name__ == "__main__":
    config = get_config()[sys.argv[1]]
    if sys.argv[1] == "bbox":
        datas = get_bbs(config["DATA_PATH"], config["SIZE"])
    elif sys.argv[1] == "ndvi":
        datas = get_ndvi(config["DATA_PATH"], config["VALID_EXT"])
    else:
        raise ValueError("Invalid argument: {}".format(sys.argv[1]))

    train_datas = {}
    valid_datas = {}

    if not os.path.exists(config["TRAIN_PATH"]):
        os.makedirs(config["TRAIN_PATH"])
    if not os.path.exists(config["VALID_PATH"]):
        os.makedirs(config["VALID_PATH"])

    # Copy and rename image files to new location
    keys_datas = list(datas.items())
    np.random.shuffle(keys_datas)
    for i, (key, data) in zip(range(len(datas)), keys_datas):
        filename = str(i) + ".jpg"
        if i < int(config["VALID_PROP"] * len(datas)):
            valid_datas[filename] = data
            shutil.copyfile(os.path.join(config["DATA_PATH"], key), os.path.join(config["VALID_PATH"], filename))
        else:
            train_datas[filename] = data
            shutil.copyfile(os.path.join(config["DATA_PATH"], key), os.path.join(config["TRAIN_PATH"], filename))

    with open(os.path.join(config["TRAIN_PATH"], "labels.json"), "w") as f:
        json.dump(train_datas, f)
    with open(os.path.join(config["VALID_PATH"], "labels.json"), "w") as f:
        json.dump(valid_datas, f)

    # Convert images and targets to np arrays
    train, train_target = data_to_array(train_datas, config["TRAIN_PATH"], target_size=config["TARGET_SIZE"])
    valid, valid_target = data_to_array(valid_datas, config["VALID_PATH"], target_size=config["TARGET_SIZE"])

    # Write arrays to disk
    bcolz.carray(train, rootdir=config["TRAIN_DATA"], mode="w").flush()
    bcolz.carray(train_target, rootdir=config["TRAIN_TARGET_DATA"], mode="w").flush()
    bcolz.carray(valid, rootdir=config["VALID_DATA"], mode="w").flush()
    bcolz.carray(valid_target, rootdir=config["VALID_TARGET_DATA"], mode="w").flush()
