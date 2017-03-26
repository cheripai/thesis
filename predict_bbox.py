import matplotlib
matplotlib.use("qt5agg")

import numpy as np
import PyQt5
import sys
from keras.preprocessing import image
from matplotlib import pyplot as plt
from cnn import VGG
from utils.utils import get_config


config = get_config()["bbox"]


def create_rect(bb, color="red"):
    return plt.Rectangle(
        (bb[2], bb[3]), bb[1], bb[0], color=color, fill=False, lw=3)


def load_img(path):
    img = image.load_img(path, target_size=TARGET_SIZE)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x


def plot(img, bb):
    plt.imshow(np.rollaxis(img, 0, 3).astype(np.uint8))
    plt.gca().add_patch(create_rect(bb))
    plt.show()


if __name__ == "__main__":

    x = load_img(sys.argv[1])

    cnn = VGG(4, config["LR"], config["DECAY"])
    cnn.model.load_weights(config["WEIGHTS_PATH"])
    bb = cnn.model.predict(x)[0]
    plot(x[0], bb)
