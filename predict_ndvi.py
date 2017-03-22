import numpy as np
import os
import sys
from keras.preprocessing import image
from inception import Inception
from vgg16 import VGG


WEIGHTS_PATH = "data/ndvi_weights.h5"
VALID_EXT = (".jpg")


def load_img(path):
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x


if __name__ == "__main__":

    cnn = Inception(4, 0.01, 0)
    cnn.model.load_weights(WEIGHTS_PATH)

    if os.path.isdir(sys.argv[1]):
        for path in os.listdir(sys.argv[1]):
            if path.endswith(VALID_EXT):
                x = load_img(os.path.join(sys.argv[1], path))
                ndvi = cnn.model.predict(x)[0]
                print(ndvi)
                
    elif sys.argv[1].endswith(VALID_EXT):
        x = load_img(sys.argv[1])
        ndvi = cnn.model.predict(x)[0]
        print(ndvi)

    else:
        print("Error: invalid argument.")
