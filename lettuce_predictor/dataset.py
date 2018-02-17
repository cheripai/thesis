import numpy as np
from PIL import Image
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import hflip, vflip


CHANNELS = 5
SIZE = 16


class RasterDataset(Dataset):
    def __init__(self, img_paths, labels, mode="valid"):

        self.img_paths = img_paths
        self.labels = labels

        self.data_transforms = {
            "train": transforms.Compose([
                transforms.Resize(SIZE),
                transforms.CenterCrop(SIZE),
                transforms.ToTensor(),
            ]),
            "valid": transforms.Compose([
                transforms.Resize(SIZE),
                transforms.CenterCrop(SIZE),
                transforms.ToTensor(),
            ]),
        }
        self.mode = mode
        self.transform = self.data_transforms[mode]

        if len(self.img_paths) != len(self.labels):
            raise Exception("Number of images and labels are mismatched!")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        arr = np.load(self.img_paths[idx])
        img = torch.zeros((CHANNELS, SIZE, SIZE))
        h_flip = random.random() < 0.5
        v_flip = random.random() < 0.5
        for i in range(CHANNELS):
            channel = Image.fromarray(arr[:, :, i] * 255).convert("L")
            if h_flip and self.mode == "train":
                channel = hflip(channel)
            if v_flip and self.mode == "train":
                channel = vflip(channel)
            img[i] = self.transform(channel)
        return img, self.labels[idx]
