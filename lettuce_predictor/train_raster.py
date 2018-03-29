import csv
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import RasterDataset
from networks import RasterNet, RasterNetPlus
from PIL import Image
from sklearn.utils import shuffle
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import hflip, vflip
from utils import normalize


CLASSES = [
    "N100IR100", "N100IR50", "N100IR25", "N100IR0", "N50IR100", "N50IR50", "N50IR25", "N50IR0",
    "N25IR100", "N25IR50", "N25IR25", "N25IR0", "N0IR100", "N0IR50", "N0IR25", "N0IR0"
]


def get_train_valid_test(root_dir, p=0.15):
    img_paths = []
    for img_dir in os.listdir(root_dir):
        img_dir = os.path.join(root_dir, img_dir)
        img_paths += [
            os.path.join(img_dir, name) for name in os.listdir(img_dir) if
            name.endswith("npy") and not "N25IR0Rep3" in name and not "N25IR50Rep3" in name or not "N25IR25Rep3" in name
        ]
    labels = [CLASSES.index(img_path.split("/")[-1].split("Rep")[0]) for img_path in img_paths]
    img_paths, labels = shuffle(img_paths, labels)
    split = int(len(img_paths) * p)
    train_set = RasterDataset(img_paths[split * 2:], labels[split * 2:], "train")
    valid_set = RasterDataset(img_paths[:split], labels[:split], "valid")
    test_set = RasterDataset(img_paths[split:split * 2], labels[split:split * 2], "valid")
    return train_set, valid_set, test_set


def correct(outputs, targets):
    _, outputs = torch.max(outputs.data, -1)
    return int(torch.sum(outputs == targets.data))


if __name__ == "__main__":
    batch_size = 32
    lr = 0.0001
    train_set, valid_set, test_set = get_train_valid_test("data/rasters")
    print("Train Images:", len(train_set))
    print("Valid Images:", len(valid_set))
    print("Test Images:", len(test_set))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    model = RasterNetPlus(16).cuda()
    criterion = nn.NLLLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=5)

    print("Learning rate: {}".format(lr))

    epochs = 100
    top_acc = 0
    for i in range(epochs):
        print("Epoch", i)
        train_loss = 0
        train_acc = 0
        model.train(True)
        for i, (X, y) in enumerate(train_loader):
            X, y = Variable(X.cuda()), Variable(y.type(torch.LongTensor).cuda())
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.data[0]
            train_acc += correct(outputs, y)
        scheduler.step(train_loss)
        train_acc /= len(train_set)
        train_loss /= i + 1
        print("Train Loss {} Train Accuracy {}".format(round(train_loss, 4), round(train_acc, 4)))

        valid_loss = 0
        valid_acc = 0
        model.train(False)
        for i, (X, y) in enumerate(valid_loader):
            X, y = Variable(X.cuda()), Variable(y.type(torch.LongTensor).cuda())
            outputs = model(X)
            loss = criterion(outputs, y)
            valid_loss += loss.data[0]
            valid_acc += correct(outputs, y)
        valid_acc /= len(valid_set)
        valid_loss /= i + 1

        if valid_acc >= top_acc:
            test_acc = 0
            top_acc = valid_acc
            for X, y in test_loader:
                X, y = Variable(X.cuda()), Variable(y.type(torch.LongTensor).cuda())
                outputs = model(X)
                test_acc += correct(outputs, y)
            test_acc /= len(test_set)
        print("Valid Loss {} Valid Accuracy {}".format(round(valid_loss, 4), round(valid_acc, 4)))

    print("Top Valid Accuracy {}".format(round(top_acc, 4)))
    print("Test Accuracy {}".format(round(test_acc, 4)))
