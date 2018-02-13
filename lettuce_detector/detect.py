import config as cfg
import json
import numpy as np
import os
import sys
import torch
from argparse import ArgumentParser
from cv2 import imread
from detectornet import DetectorNet
from PIL import Image
from skimage import color
from torch.autograd import Variable
from utils import pyramid, sliding_window, non_max_suppression


def detect(model, image):
    j = 0
    boxes = []
    cur_boxes = []
    windows = Variable(torch.zeros((cfg.batch_size, 3, cfg.img_size, cfg.img_size)))

    if cfg.use_cuda:
        model = model.cuda()
        windows = windows.cuda()

    # Slides window across image running classifier on each window
    # for i, resized in enumerate(pyramid(image, cfg.scale_amt)):
    for (x, y, window) in sliding_window(image, cfg.step_size, window_size=(cfg.win_width, cfg.win_height)):
        if window.shape[0] != cfg.win_height or window.shape[1] != cfg.win_width:
            continue

        window = cfg.data_transforms["valid"](Image.fromarray(window))
        if cfg.use_cuda: window = window.cuda()

        windows[j] = window
        j += 1

        # cur_boxes.append((x, y, x + cfg.win_width * cfg.scale_amt**i, y + cfg.win_height * cfg.scale_amt**i))
        cur_boxes.append((x, y, x + cfg.win_width, y + cfg.win_height))

        if j == cfg.batch_size:
            _, predictions = model(windows).data.topk(1)
            predictions_indexes = np.where(predictions.cpu().numpy().ravel() == 1)[0]
            boxes += list(cur_boxes[i] for i in predictions_indexes)
            cur_boxes = []
            j = 0

    # Remove redundant boxes
    boxes = non_max_suppression(boxes, cfg.overlap_thresh)
    return sorted(boxes, key=lambda box: box[0])


def boxes_to_json(boxes, image_path):
    return {
        "class": "image",
        "filename": image_path,
        "annotations": [{
            "class": "rect",
            "height": int(y2 - y1),
            "width": int(x2 - x1),
            "x": int(x1),
            "y": int(y1)
        } for (x1, y1, x2, y2) in boxes]
    }


def img_path_to_annotation(model, image_path):
    image = imread(image_path)
    boxes = detect(model, image)
    return boxes_to_json(boxes, os.path.abspath(image_path))


if __name__ == "__main__":
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    try:
        model = DetectorNet(2)
        model.load_state_dict(torch.load(cfg.weights_path))
    except:
        print("Error: Model not found at '{}'".format(cfg.model_path))
        print("Check the path in config.py or run train.py to generate a model")
    
    if os.path.isdir(in_path):
        annotations = []
        for image_path in os.listdir(in_path):
            image_path = os.path.join(in_path, image_path)
            annotations.append(img_path_to_annotation(model, image_path))
        annotations = sorted(annotations, key=lambda annotation: annotation["filename"])
        with open(out_path, "w") as f:
            f.write(json.dumps(annotations))
    else:
        annotation = img_path_to_annotation(model, in_path)
        with open(out_path, "w") as f:
            f.write(json.dumps([annotation]))
