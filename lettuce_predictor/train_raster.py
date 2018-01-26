import csv
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from networks import DenseNet, RasterNet
from PIL import Image
from sklearn.utils import shuffle
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import hflip, vflip


CLASSES = ["IR100", "IR50", "IR25", "IR0"]
CHANNELS = 5
SIZE = 16


class TreatmentDataset(Dataset):
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
            channel = Image.fromarray(arr[:, :, i]*255).convert("L")
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

 
def get_train_test_csv(csv_dir, target_col, p=0.15):
    img_paths = []
    labels = []
    for csv_path in os.listdir(csv_dir):
        if csv_path.endswith(".csv"):
            with open(os.path.join(csv_dir,csv_path)) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        labels.append(float(row[target_col]))
                        img_paths.append(row["img_path"])
                    except:
                        continue
    labels = normalize(labels)
    mean = np.mean(labels)
    labels = [int(label < mean) for label in labels]
    split = int(p * len(img_paths))
    return TreatmentDataset(img_paths[split:], labels[split:], mode="train"), TreatmentDataset(img_paths[:split], labels[:split])


def correct(outputs, targets):
    _, outputs = torch.max(outputs.data, -1)
    return torch.sum(outputs == targets.data)

        
if __name__ == "__main__":
    batch_size = 32
    lr = 0.0001
    leaf_train, leaf_valid = get_train_test_csv("data/rasters_csv", "Average Height")
    print("Train Images:", len(leaf_train))
    print("Valid Images:", len(leaf_valid))
    train_loader = DataLoader(leaf_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(leaf_valid, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = RasterNet(len(CLASSES)).cuda()
    criterion = nn.NLLLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=5)

    print("Learning rate: {}".format(lr))

    epochs = 100
    for i in range(epochs):
        print("Epoch", i)
        train_loss = 0
        train_acc = 0
        count = 0
        for X, y in train_loader:
            X, y = Variable(X.cuda()), Variable(y.type(torch.LongTensor).cuda())
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.data[0]
            train_acc += correct(outputs ,y)
            count += X.size(0)
        train_acc /= count
        scheduler.step(train_loss)
        print("Train Loss {} Train Accuracy {}".format(round(train_loss, 4), round(train_acc, 4)))

        valid_loss = 0
        valid_acc = 0
        count = 0
        for X, y in valid_loader:
            X, y = Variable(X.cuda()), Variable(y.type(torch.LongTensor).cuda())
            outputs = model(X)
            loss = criterion(outputs, y)
            valid_loss += loss.data[0]
            valid_acc += correct(outputs ,y)
            count += X.size(0)
        valid_acc /= count
        print("Valid Loss {} Valid Accuracy {}".format(round(valid_loss, 4), round(valid_acc, 4)))
