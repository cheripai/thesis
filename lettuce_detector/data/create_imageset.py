import os
import sys
from random import shuffle

VALID_PROP = 0.15

if __name__ == "__main__":
    images_path = sys.argv[1]
    imageset_path = sys.argv[2]

    if not os.path.exists(imageset_path):
        os.makedirs(imageset_path)

    for root, dirs, files in os.walk(images_path):
        fpaths = [os.path.join(root.split("/")[-1], f).split(".")[0] for f in files]   

    shuffle(fpaths)
    
    p = int(len(fpaths) * VALID_PROP)
    valid_fpaths = fpaths[:p]
    train_fpaths = fpaths[p:]

    with open(os.path.join(imageset_path, "valid.txt"), "w+") as f:
        f.writelines("\n".join(valid_fpaths))
    with open(os.path.join(imageset_path, "train.txt"), "w+") as f:
        f.writelines("\n".join(train_fpaths))
