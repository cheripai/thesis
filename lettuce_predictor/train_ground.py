import csv
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from networks import DenseNet, ResNet, VGG
from PIL import Image
from sklearn.utils import shuffle
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class GroundDataset(Dataset):
    def __init__(self, img_paths, labels, mode="valid"):
        
        self.data_transforms = { 
            "train": transforms.Compose([
                transforms.Resize(240),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            "valid": transforms.Compose([
                transforms.Resize(240),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
        }

        self.transform = self.data_transforms[mode]
        self.img_paths = img_paths
        self.labels = labels

        if len(img_paths) != len(labels):
            raise Exception("Number of images and labels are mismatched!")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx])
        img = img.convert("RGB")
        img = self.transform(img)
        return img, self.labels[idx]


def normalize(xs):
    """ Puts target data in 0 to 1 range.
    """
    min_x = min(xs)
    max_x = max(xs)
    return [(x - min_x) / (max_x - min_x) for x in xs]


def get_train_test(csv_dir, target_col, p=0.15):
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
    return GroundDataset(img_paths[split:], labels[split:], mode="train"), GroundDataset(img_paths[:split], labels[:split])


def correct(outputs, targets):
    _, outputs = torch.max(outputs.data, -1)
    return torch.sum(outputs == targets.data)

        
if __name__ == "__main__":
    batch_size = 8
    lr = 0.00001
    train_set, valid_set = get_train_test("data/ground_data", "WP")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print("Train Images: {}".format(len(train_set)))
    print("Valid Images: {}".format(len(valid_set)))

    model = DenseNet(2, mode="classification", p=0.05).cuda()
    # model = ResNet(2, mode="classification", p=0.15).cuda()
    # model = VGG(2, p=0.15).cuda()
    criterion = nn.NLLLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Learning rate: {}".format(lr))

    epochs = 60
    for i in range(epochs):
        print("Epoch", i)
        train_loss = 0
        train_acc = 0
        count = 0
        for j, (X, y) in enumerate(train_loader):
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
        print("Train Loss {} Train Accuracy {}".format(round(train_loss, 4), round(train_acc, 4)))

        valid_loss = 0
        valid_acc = 0
        count = 0
        for j, (X, y) in enumerate(valid_loader):
            X, y = Variable(X.cuda()), Variable(y.type(torch.LongTensor).cuda())
            outputs = model(X)
            loss = criterion(outputs, y)
            valid_loss += loss.data[0]
            valid_acc += correct(outputs ,y)
            count += X.size(0)
        valid_acc /= count
        print("Valid Loss {} Valid Accuracy {}".format(round(valid_loss, 4), round(valid_acc, 4)))
