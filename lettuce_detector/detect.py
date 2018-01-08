import config as cfg
import sys
import torch
from argparse import ArgumentParser
from cv2 import imread, imwrite, rectangle
from models.simplenet import SimpleNet
from os import path
from PIL import Image
from skimage import color
from torch.autograd import Variable
from utils import pyramid, sliding_window, non_max_suppression

import random
import string

if __name__ == "__main__":
    image_path = sys.argv[1]
    image = imread(image_path)
    try:
        model = SimpleNet(2)
        model.load_state_dict(torch.load(cfg.weights_path))
    except:
        print("Error: Model not found at '{}'".format(cfg.model_path))
        print("Check the path in config.py or run train.py to generate a model")

    if cfg.use_cuda: model = model.cuda()

    boxes = []
    # Slides window across image running classifier on each window
    for i, resized in enumerate(pyramid(image, cfg.scale_amt)):
        for (x, y, window) in sliding_window(resized, cfg.step_size, window_size=(cfg.win_width, cfg.win_height)):
            if window.shape[0] != cfg.win_height or window.shape[1] != cfg.win_width:
                continue

            # imwrite("tmp/" +''.join(random.choices(string.ascii_uppercase + string.digits, k=4))+".jpg", window)
            # TODO: Load in batches
            X = Variable(cfg.data_transforms["valid"](Image.fromarray(window)))
            X = X.unsqueeze(0)
            if cfg.use_cuda: X = X.cuda()

            _, prediction = model(X).data.topk(1)
            if int(prediction) == 1:
                boxes.append((x, y, x + cfg.win_width * cfg.scale_amt**i, y + cfg.win_height * cfg.scale_amt**i))

    # Remove redundant boxes
    boxes = non_max_suppression(boxes, cfg.overlap_thresh)

    # TODO: Write json file for bounding boxes so user can use sloth to fix
    for (x1, y1, x2, y2) in boxes:
        rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    old_path = path.split(image_path)[1].split(".")
    new_path = old_path[0] + "_boxed." + old_path[1]
    imwrite(new_path, image)
    print("Saved as {}".format(new_path))