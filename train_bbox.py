import json
import numpy as np
import os
from keras.layers import Dense
from keras.preprocessing import image
from vgg16 import VGG16



TRAIN_PATH = "data/train_bbox"
VALID_PATH = "data/valid_bbox"
SIZE = (4000, 3000)
TARGET_SIZE = (224, 224)


def create_rect(bb, color="red"):
    return plt.Rectangle((bb[2], bb[3]), bb[1], bb[0], color=color, fill=False, lw=3)


if __name__ == "__main__":
    train_bbs = json.load(open(os.path.join(TRAIN_PATH, "annotations.json")))
    valid_bbs = json.load(open(os.path.join(VALID_PATH, "annotations.json")))

    train = np.zeros((len(train_bbs), 3, TARGET_SIZE[0], TARGET_SIZE[1]))
    valid = np.zeros((len(valid_bbs), 3, TARGET_SIZE[0], TARGET_SIZE[1]))
    train_target, valid_target = [], []
    for i, (key, bb) in zip(range(len(train_bbs)), train_bbs.items()):
        img = image.load_img(os.path.join(TRAIN_PATH, key), target_size=TARGET_SIZE)
        train[i] = image.img_to_array(img)
        train_target.append(bb)
    for i, (key, bb) in zip(range(len(valid_bbs)), valid_bbs.items()):
        img = image.load_img(os.path.join(VALID_PATH, key), target_size=TARGET_SIZE)
        valid[i] = image.img_to_array(img)
        valid_target.append(bb)

    vgg = VGG16()
    vgg.model.pop()
    vgg.model.add(Dense(4))
    vgg.model.compile(optimizer="Adam", loss="mse")
    vgg.model.fit(train, train_target, batch_size=32, nb_epoch=30, validation_data=(valid, valid_target))
