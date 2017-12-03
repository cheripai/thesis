import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.utils import shuffle
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


class LeafDataset(Dataset):
    def __init__(self, img_paths, labels):
        
        self.img_paths = img_paths
        self.labels = labels

        if len(self.img_paths) != len(self.labels):
            raise Exception("Number of images and labels are mismatched!")

        self.transform = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx])
        img = img.convert("RGB")
        img = self.transform(img)
        return img, self.labels[idx]


def get_train_test(img_dir, label_file, p=0.2):
        img_paths = [os.path.join(img_dir, name) for name in os.listdir(img_dir)]
        labels, missing_idx = load_labels(label_file)
        for i in sorted(missing_idx, reverse=True):
            del img_paths[i]
        # labels = normalize(labels)
        img_paths, labels = shuffle(img_paths, labels)
        split = int(len(img_paths) * p)
        return LeafDataset(img_paths[split:], labels[split:]), LeafDataset(img_paths[:split], labels[:split])


def normalize(data):
    min_val = min(data)
    max_val = max(data)
    return [(x - min_val) / (max_val - min_val) for x in data]


def load_labels(label_filename):
    labels = []
    missing_idx = []
    with open(label_filename) as f:
        for i, line in enumerate(f.readlines()):
            try:
                labels.append(int(line.strip()))
            except ValueError:
                missing_idx.append(i)
    return labels, missing_idx
            

class DenseNet(nn.Module):
    def __init__(self, features=128, p=0.1):
        super(DenseNet, self).__init__()
        self.model = models.densenet121(pretrained=True).cuda()
        self.model.classifier = nn.Linear(self.model.classifier.in_features, features)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(features)
        self.dropout = nn.Dropout(p=p)
        self.dense = nn.Linear(features, 3)
        self.softmax = nn.Softmax()

    def forward(self, X):
        output = self.dropout(self.bn(self.relu(self.model(X))))
        return self.softmax(self.dense(output))


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, 3, padding=1)
        self.fc1 = nn.Linear(256* 7 * 7, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)
        x = F.max_pool2d(F.relu(self.conv5(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return F.sigmoid(x)

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def correct(outputs, targets):
    _, outputs = torch.max(outputs.data, -1)
    return torch.sum(outputs == targets.data)

        
if __name__ == "__main__":
    batch_size = 24
    lr = 0.005
    leaf_train, leaf_valid = get_train_test("data/img", "data/chlorophyll_classes.txt")
    train_loader = DataLoader(leaf_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(leaf_valid, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = DenseNet().cuda()
    # criterion = nn.MSELoss()
    criterion = nn.NLLLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(model)
    print("Learning rate: {}".format(lr))

    epochs = 20
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
        valid_acc = valid_acc / count
        print("Valid Loss {} Valid Accuracy {}".format(round(valid_loss, 4), round(valid_acc, 4)))
