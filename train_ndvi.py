import bcolz
import sys
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from os import path
from models.cnn import Inception, ResNet, VGG
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
            width_shift_range=config["WIDTH_SHIFT_RANGE"],
            height_shift_range=config["HEIGHT_SHIFT_RANGE"],
            zoom_range=config["ZOOM_RANGE"])
    else:
        print("No training or validation data found.")
        print("Run make_dataset.py first!")
        sys.exit(1)

    cnn = Inception(1, config["LR"], config["DECAY"])
    callbacks = [
        ModelCheckpoint(config["WEIGHTS_PATH"], monitor="val_loss", save_best_only=True, save_weights_only=True),
        TensorBoard(log_dir="results/logs")
    ]
    cnn.model.fit_generator(
        datagen.flow(train, train_target, batch_size=config["BATCH_SIZE"]),
        len(train) // config["BATCH_SIZE"] + 1,
        epochs=config["EPOCHS"],
        validation_data=(valid, valid_target))

    cnn.model.save_weights(config["WEIGHTS_PATH"])
