import numpy as np
import os
import sys
from keras.preprocessing import image
from cnn import Inception
from utils.utils import get_config


config = get_config()["ndvi"]


def load_img(path):
    img = image.load_img(path, target_size=config["TARGET_SIZE"])
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x


if __name__ == "__main__":

    cnn = Inception(1, config["LR"], config["DECAY"])
    cnn.model.load_weights(config["WEIGHTS_PATH"])

    if os.path.isdir(sys.argv[1]):
        for path in os.listdir(sys.argv[1]):
            if path.endswith(config["VALID_EXT"]):
                x = load_img(os.path.join(sys.argv[1], path))
                ndvi = cnn.model.predict(x)[0]
                print(ndvi)
                
    elif sys.argv[1].endswith(config["VALID_EXT"]):
        x = load_img(sys.argv[1])
        ndvi = cnn.model.predict(x)[0]
        print(ndvi)

    else:
        print("Error: invalid argument.")
