import config as c
import math
import numpy as np
import torch
import torch.nn as nn
import sys
from imageloader import PairedImageLoader
from models.fcdensenet import FCDenseNet
from PIL import Image
from scipy.misc import imsave
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms


def flatten_to_tiles(img, tile_size=c.IMG_SIZE):
    h, w, ch = img.shape
    tiles = np.zeros(((h // tile_size) * (w // tile_size), tile_size, tile_size, ch))
    for i in range(h // tile_size - 1):
        y1, y2 = i * tile_size, (i+1) * tile_size
        for j in range(w // tile_size - 1):
            x1, x2 = j * tile_size, (j+1) * tile_size
            tiles[i * (w // tile_size) + j] = img[y1:y2, x1:x2]
    return tiles


def unflatten_tiles(tiles, h, w):
    img = np.zeros((h, w))
    tile_size = tiles.shape[-1]
    tiles_per_row = w // tile_size
    for i in range(tiles.shape[0]):
        row = (i // tiles_per_row) * tile_size
        col = (i % tiles_per_row) * tile_size
        img[row:row+tile_size, col:col+tile_size] = tiles[i]
    return img


if __name__ == "__main__":
    img = np.asarray(Image.open(sys.argv[1]), dtype=np.uint8)[:,:,:3]
    h, w, ch = img.shape

    padded_img = np.zeros((h + 2*c.IMG_SIZE, w + 2*c.IMG_SIZE, ch))
    padded_img[:h, :w, :] = img
    p_h, p_w, _ = padded_img.shape

    model = FCDenseNet([4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4], 16, drop_rate=0.0, n_classes=1).cuda()
    model.load_state_dict(torch.load(c.WEIGHTS_PATH))
    model.train(False)

    output_map = np.zeros((h, w))

    in_tiles = np.moveaxis(flatten_to_tiles(padded_img, c.IMG_SIZE), -1, 1)
    out_tiles = np.zeros((in_tiles.shape[0], c.IMG_SIZE, c.IMG_SIZE))
    for i in range(in_tiles.shape[0] // c.BATCH_SIZE):
        if (i+1) * c.BATCH_SIZE > in_tiles.shape[0]:
            batch_size = in_tiles.shape[0] % c.BATCH_SIZE
        else:
            batch_size = c.BATCH_SIZE

        in_segment = Variable(torch.from_numpy(in_tiles[i*c.BATCH_SIZE:(i+1)*batch_size] / 256).type(torch.FloatTensor)).cuda()
        out_segment = model(in_segment).data.squeeze().cpu().numpy()
        out_tiles[i*c.BATCH_SIZE:(i+1)*batch_size] = out_segment

    out_img = unflatten_tiles(out_tiles, p_h, p_w)
    imsave("output.png", out_img[:h, :w])
