import os
import shutil
import string
import sys
import random
from sklearn.utils import shuffle


VALID_PROP = 0.15
X_VALID_DIR = "x_valid"
Y_VALID_DIR = "y_valid"
X_TRAIN_DIR = "x_train"
Y_TRAIN_DIR = "y_train"


def random_name(n=6):
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=n))


if __name__ == "__main__":
    x_dir = sys.argv[1]
    y_dir = sys.argv[2]

    x_paths = [os.path.join(x_dir, f) for f in os.listdir(x_dir)]
    y_paths = [os.path.join(y_dir, f) for f in os.listdir(y_dir)]

    x_paths, y_paths = shuffle(x_paths, y_paths)

    split_index = int(len(x_paths) * VALID_PROP)
    x_valid = x_paths[:split_index]
    y_valid = y_paths[:split_index]
    x_train = x_paths[split_index:]
    y_train = y_paths[split_index:]

    if not os.path.exists(X_VALID_DIR):
        os.makedirs(X_VALID_DIR)
    if not os.path.exists(Y_VALID_DIR):
        os.makedirs(Y_VALID_DIR)
    if not os.path.exists(X_TRAIN_DIR):
        os.makedirs(X_TRAIN_DIR)
    if not os.path.exists(Y_TRAIN_DIR):
        os.makedirs(Y_TRAIN_DIR)

    for i in range(len(x_valid)):
        name = random_name() + ".png"
        while os.path.isfile(os.path.join(X_VALID_DIR, name)):
            name = random_name()
        shutil.copyfile(x_valid[i], os.path.join(X_VALID_DIR, name))
        shutil.copyfile(y_valid[i], os.path.join(Y_VALID_DIR, name))

    for i in range(len(x_train)):
        name = random_name() + ".png"
        while os.path.isfile(os.path.join(X_TRAIN_DIR, name)):
            name = random_name()
        shutil.copyfile(x_train[i], os.path.join(X_TRAIN_DIR, name))
        shutil.copyfile(y_train[i], os.path.join(Y_TRAIN_DIR, name))
