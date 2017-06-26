import bcolz
import sys
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from os import path
from models.cnn import Inception, ResNet, VGG
from utils.utils import get_config


config = get_config()["ir"]


if __name__ == "__main__":

    if path.exists(config["TRAIN_PATH"]) and path.exists(config["VALID_PATH"]):
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            horizontal_flip=True,
            width_shift_range=config["WIDTH_SHIFT_RANGE"],
            height_shift_range=config["HEIGHT_SHIFT_RANGE"],
            zoom_range=config["ZOOM_RANGE"])
        test_datagen = ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow_from_directory(
            config["TRAIN_PATH"],
            target_size=config["TARGET_SIZE"],
            batch_size=config["BATCH_SIZE"],
            class_mode="categorical")
        validation_generator = test_datagen.flow_from_directory(
            config["VALID_PATH"],
            target_size=config["TARGET_SIZE"],
            batch_size=config["BATCH_SIZE"],
            class_mode="categorical")
    else:
        print("No training or validation data found.")
        sys.exit(1)

    cnn = Inception(4, config["LR"], config["DECAY"], dropout=0, classification=True)
    callbacks = [
        ModelCheckpoint(config["WEIGHTS_PATH"], monitor="val_loss", save_best_only=True, save_weights_only=True),
        TensorBoard(log_dir="results/logs")
    ]
    cnn.model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.n // config["BATCH_SIZE"] + 1,
        epochs=config["EPOCHS"],
        validation_data=validation_generator,
        validation_steps=validation_generator.n // config["BATCH_SIZE"] + 1)

    cnn.model.save_weights(config["WEIGHTS_PATH"])
