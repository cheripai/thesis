import bcolz
import sys
from keras.preprocessing.image import ImageDataGenerator
from os import path
from cnn import Inception, ResNet, VGG
from utils.utils import get_config


config = get_config()["ndvi"]


if __name__ == "__main__":

    if path.exists(config["TRAIN_DATA"]) and path.exists(config["VALID_DATA"]) \
        and path.exists(config["TRAIN_TARGET_DATA"]) and path.exists(config["VALID_TARGET_DATA"]):
        train = bcolz.open(config["TRAIN_DATA"])[:]
        train_target = bcolz.open(config["TRAIN_TARGET_DATA"])[:]
        valid = bcolz.open(config["VALID_DATA"])[:]
        valid_target = bcolz.open(config["VALID_TARGET_DATA"])[:]
        
        datagen = ImageDataGenerator(
            horizontal_flip=True,
            width_shift_range=0.05,
            height_shift_range=0.05,
            zoom_range=0.05)
    else:
        print("No training or validation data found.")
        print("Run make_dataset.py first!")
        sys.exit(1)

    cnn = Inception(1, config["LR"], config["DECAY"])
    cnn.model.fit_generator(
        datagen.flow(train, train_target, batch_size=config["BATCH_SIZE"]),
        len(train),
        epochs=config["EPOCHS"],
        validation_data=(valid, valid_target))

    cnn.model.save_weights(config["WEIGHTS_PATH"])
