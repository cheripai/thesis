import bcolz
import sys
from os import path
from inception import Inception
from vgg16 import VGG


TRAIN_DATA = "data/train_ndvi/train.dat"
TRAIN_TARGET_DATA = "data/train_ndvi/train_target.dat"
VALID_DATA = "data/valid_ndvi/valid.dat"
VALID_TARGET_DATA = "data/valid_ndvi/valid_target.dat"
WEIGHTS_PATH = "data/ndvi_weights.h5"


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

    cnn = Inception(1, 0.01, 0)
    cnn.model.fit(
        train,
        train_target,
        batch_size=64,
        nb_epoch=100,
        validation_data=(valid, valid_target))

    cnn.model.save_weights(WEIGHTS_PATH)
