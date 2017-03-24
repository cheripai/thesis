import bcolz
import sys
from keras.preprocessing.image import ImageDataGenerator
from os import path
from cnn import Inception, ResNet, VGG


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
        train = train.reshape(train.shape[0], train.shape[2], train.shape[3], train.shape[1])
        valid = valid.reshape(valid.shape[0], valid.shape[2], valid.shape[3], valid.shape[1])
        
        datagen = ImageDataGenerator(
            horizontal_flip=True,
            width_shift_range=0.05,
            height_shift_range=0.05,
            zoom_range=0.05)
    else:
        print("No training or validation data found.")
        print("Run make_dataset.py first!")
        sys.exit(1)

    cnn = Inception(1, 0.01, 0)
    cnn.model.fit_generator(
        datagen.flow(train, train_target, batch_size=64),
        len(train),
        epochs=100,
        validation_data=(valid, valid_target))

    cnn.model.save_weights(WEIGHTS_PATH)
