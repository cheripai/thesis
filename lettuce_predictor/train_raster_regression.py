import csv
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from networks import RasterNet, RasterNetPlus
from PIL import Image
from sklearn.utils import shuffle
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
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


def normalize(xs):
    """ Puts target data in 0 to 1 range.
    """
    min_x = min(xs)
    max_x = max(xs)
    return [(x - min_x) / (max_x - min_x) for x in xs]


def get_train_valid_test_csv(csv_dir, target_col, p=0.15):
    img_paths = []
    labels = []
    for csv_path in os.listdir(csv_dir):
        if csv_path.endswith(".csv"):
            with open(os.path.join(csv_dir, csv_path)) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        labels.append(float(row[target_col]))
                        img_paths.append(row["img_path"])
                    except:
                        continue

    img_paths, labels = shuffle(img_paths, labels)
    labels = normalize(labels)

    split = int(p * len(img_paths))
    train_set = RasterDataset(img_paths[split * 2:], labels[split * 2:], "train")
    valid_set = RasterDataset(img_paths[:split], labels[:split], "valid")
    test_set = RasterDataset(img_paths[split:split * 2], labels[split:split * 2], "valid")
    return train_set, valid_set, test_set


if __name__ == "__main__":
    batch_size = 32
    lr = 0.0001
    train_set, valid_set, test_set = get_train_valid_test_csv("data/rasters_csv", "Chlorophyll")
    print("Train Images:", len(train_set))
    print("Valid Images:", len(valid_set))
    print("Test Images:", len(test_set))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    model = RasterNet(1, mode="regression").cuda()
    criterion = nn.MSELoss().cuda()
    mae = nn.L1Loss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=5)

    print("Learning rate: {}".format(lr))

    epochs = 100
    top_acc = 0
    for i in range(epochs):
        print("Epoch", i)
        train_loss = 0
        model.train(True)
        for i, (X, y) in enumerate(train_loader):
            X, y = Variable(X.cuda()), Variable(y.type(torch.FloatTensor).cuda())
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.data[0]
        scheduler.step(train_loss)
        train_loss /= i + 1
        print("Train MSE {}".format(round(train_loss, 4)))

        valid_loss = 0
        test_loss = 0
        best_loss = 1e10
        model.train(False)
        for i, (X, y) in enumerate(valid_loader):
            X, y = Variable(X.cuda()), Variable(y.type(torch.FloatTensor).cuda())
            outputs = model(X)
            loss = mae(outputs, y)
            valid_loss += loss.data[0]
        valid_loss /= i + 1

        if valid_loss <= best_loss:
            best_loss = valid_loss
            test_loss = 0
            for X, y in test_loader:
                X, y = Variable(X.cuda()), Variable(y.type(torch.FloatTensor).cuda())
                outputs = model(X)
                loss = mae(outputs, y)
                test_loss += loss.data[0]
            test_loss /= len(test_set)
        print("Valid MAE {}".format(round(valid_loss, 4)))

    print("Best Valid MAE {}".format(round(best_loss, 4)))
    print("Test MAE {}".format(round(test_loss, 4)))
