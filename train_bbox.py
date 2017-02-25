import bcolz
import sys
from os import path
from vgg16 import VGG


TRAIN_DATA = "data/train_bbox/train.dat"
TRAIN_TARGET_DATA = "data/train_bbox/train_target.dat"
VALID_DATA = "data/valid_bbox/valid.dat"
VALID_TARGET_DATA = "data/valid_bbox/valid_target.dat"
WEIGHTS_PATH = "data/vgg_bbox.h5"


if __name__ == "__main__":

    if path.exists(TRAIN_DATA) and path.exists(VALID_DATA) \
        and path.exists(TRAIN_TARGET_DATA) and path.exists(VALID_TARGET_DATA):
        train = bcolz.open(TRAIN_DATA)[:]
        train_target = bcolz.open(TRAIN_TARGET_DATA)[:]
        valid = bcolz.open(VALID_DATA)[:]
        valid_target = bcolz.open(VALID_TARGET_DATA)[:]
    else:
        print("No training or validation data found.")
        print("Run make_bbox_dataset.py first!")
        sys.exit(1)

    vgg = VGG(4, 0.01, 0)
    vgg.model.fit(
        train,
        train_target,
        batch_size=64,
        nb_epoch=100,
        validation_data=(valid, valid_target))

    vgg.model.save_weights(WEIGHTS_PATH)
