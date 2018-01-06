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


#CLASSES = ["N100IR100", "N100IR50", "N100IR25", "N100IR0",
#           "N50IR100", "N50IR50", "N50IR25", "N50IR0",
#           "N25IR100", "N25IR50", "N25IR25", "N25IR0",
#           "N0IR100", "N0IR50", "N0IR25", "N0IR0"]
CLASSES = ["IR100", "IR50", "IR25", "IR0"]


class TreatmentDataset(Dataset):
    def __init__(self, img_paths, labels, mode="valid"):
        
        self.img_paths = img_paths
        self.labels = labels

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

        if len(self.img_paths) != len(self.labels):
            raise Exception("Number of images and labels are mismatched!")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx])
        img = img.convert("RGB")
        img = self.transform(img)
        return img, self.labels[idx]


def get_train_test(img_dir, p=0.2):
        img_paths = [os.path.join(img_dir, name) for name in os.listdir(img_dir) if name.endswith("png")]
        labels = [CLASSES.index("IR"+img_path.split("IR")[-1].split("Rep")[0]) for img_path in img_paths]
        img_paths, labels = shuffle(img_paths, labels)
        split = int(len(img_paths) * p)
        return TreatmentDataset(img_paths[split:], labels[split:], "train"), TreatmentDataset(img_paths[:split], labels[:split], "valid")


class DenseNet(nn.Module):
    def __init__(self, outputs, p=0.1):
        super(DenseNet, self).__init__()
        self.model = models.densenet161(pretrained=True, drop_rate=p).cuda()
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(self.model.classifier.in_features, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=p),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, outputs),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, X):
        return self.model(X)


class ResNet(nn.Module):
    def __init__(self, outputs, p=0.1):
        super(ResNet, self).__init__()
        self.model = models.resnet50(pretrained=True).cuda()
        self.model.fc = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(self.model.fc.in_features, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=p),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Linear(128, outputs),
            nn.LogSoftmax(dim=-1),
        )
            
    def forward(self, X):
        return self.model(X)


class VGG(nn.Module):
    def __init__(self, outputs, p=0.1):
        super(VGG, self).__init__()
        self.model = models.vgg16_bn(pretrained=True).features
        self.classifier = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(4096),
            nn.Dropout(p=p),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, outputs),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, X):
        output = self.model(X)
        output = output.view(output.size(0), -1)
        return self.classifier(output)


def correct(outputs, targets):
    _, outputs = torch.max(outputs.data, -1)
    return torch.sum(outputs == targets.data)

        
if __name__ == "__main__":
    batch_size = 8
    lr = 0.0001
    leaf_train, leaf_valid = get_train_test("data/img")
    train_loader = DataLoader(leaf_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(leaf_valid, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = DenseNet(len(CLASSES), p=0.15).cuda()
    # model = VGG(len(CLASSES), p=0.15).cuda()
    # model = ResNet(len(CLASSES), p=0.2).cuda()
    criterion = nn.NLLLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Learning rate: {}".format(lr))

    epochs = 60
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
        valid_acc /= count
        print("Valid Loss {} Valid Accuracy {}".format(round(valid_loss, 4), round(valid_acc, 4)))
