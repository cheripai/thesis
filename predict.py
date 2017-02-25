import matplotlib
matplotlib.use("qt5agg")

import numpy as np
import PyQt5
import sys
from keras.preprocessing import image
from matplotlib import pyplot as plt
from vgg16 import VGG


WEIGHTS_PATH = "data/vgg_bbox.h5"


def create_rect(bb, color="red"):
    return plt.Rectangle(
        (bb[2], bb[3]), bb[1], bb[0], color=color, fill=False, lw=3)


def load_img(path):
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x


def plot(img, bb):
    plt.imshow(np.rollaxis(img, 0, 3).astype(np.uint8))
    plt.gca().add_patch(create_rect(bb))
    plt.show()


if __name__ == "__main__":

    x = load_img(sys.argv[1])

    vgg = VGG(4, 0.01, 0)
    vgg.model.load_weights(WEIGHTS_PATH)
    bb = vgg.model.predict(x)[0]
    plot(x[0], bb)
