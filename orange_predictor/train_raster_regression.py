import csv
import numpy as np
import os
from PIL import Image
import random
import torch
import torch.nn as nn
import torch.optim as optim
from networks import RasterNet, RasterNet3, SqueezeNet
from sklearn.utils import shuffle
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import hflip, vflip
from utils import normalize, save_error_chart


CHANNELS = 3
SIZE = 96

# These have bad crops
EXCLUDE_ID = ("26D", "35E", "42C", "44F")

class RasterDataset(Dataset):
    def __init__(self, img_paths, labels, mode="valid"):

        self.img_paths = img_paths
        self.labels = labels

        self.data_transforms = {
            "train": transforms.Compose([
                transforms.Resize((SIZE+4, SIZE+4)),
                transforms.RandomCrop(SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
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
        arr = arr[:,:,:CHANNELS]
        arr = arr[10:-10, 10:-10, :]
        img = Image.fromarray(np.uint8(arr*255))
        img = img.convert("RGB")
        img = self.transform(img)
        return img, self.labels[idx]


def get_train_valid_test_csv(csv_path, img_dir, target_col, p=0.15):
    img_paths = []
    labels = []
    with open(os.path.join(csv_path)) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["ID"] in EXCLUDE_ID:
                continue
            try:
                labels.append(float(row[target_col]))
                img_paths.append(os.path.join(img_dir, row["ID"]+".npy"))
            except:
                continue

    img_paths, labels = shuffle(img_paths, labels)
    labels = normalize(labels)
    split = int(p * len(img_paths))
    return RasterDataset(img_paths[split*2:], labels[split*2:], "train"), RasterDataset(img_paths[:split], labels[:split], "valid"), RasterDataset(img_paths[split:split*2], labels[split:split*2], "valid")


if __name__ == "__main__":
    batch_size = 32
    lr = 0.0005
    value_name = "NDVI"
    train_set, valid_set, test_set = get_train_valid_test_csv("data/rasters_csv/data.csv", "data/rasters_csv/2017-11-17", value_name)
    print("Train Images:", len(train_set))
    print("Valid Images:", len(valid_set))
    print("Test Images:", len(test_set))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    model = RasterNet3(1, mode="regression").cuda()
    criterion = nn.MSELoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=5)

    print("Learning rate: {}".format(lr))

    epochs = 100
    test_loss = 0
    best_loss = 1e10
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
        print("Train Loss {}".format(round(np.sqrt(train_loss), 4)))

        valid_loss = 0
        model.train(False)
        for i, (X, y) in enumerate(valid_loader):
            X, y = Variable(X.cuda()), Variable(y.type(torch.FloatTensor).cuda())
            outputs = model(X)
            loss = criterion(outputs, y)
            valid_loss += loss.data[0]
        valid_loss /= i + 1

        if valid_loss <= best_loss:
            best_loss = valid_loss
            test_loss = 0
            ys = np.zeros(len(test_set))
            yhats = np.zeros(len(test_set))
            for i, (X, y) in enumerate(test_loader):
                X, y = Variable(X.cuda()), Variable(y.type(torch.FloatTensor).cuda())
                outputs = model(X)
                loss = criterion(outputs, y)
                test_loss += loss.data[0]
                ys[i] = y.data[0]
                yhats[i] = outputs.data[0, 0]
            test_loss /= len(test_set)
            save_error_chart(ys, yhats, value_name, "results/{}.png".format(value_name))
        print("Valid Loss {}".format(round(np.sqrt(valid_loss), 4)))

    print("Best Valid Loss {}".format(round(np.sqrt(best_loss), 4)))
    print("Test Loss {}".format(round(np.sqrt(test_loss), 4)))
