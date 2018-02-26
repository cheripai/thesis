import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SqueezeNet(nn.Module):
    def __init__(self, outputs, mode="classification", p=0.1):
        super(SqueezeNet, self).__init__()
        self.mode = mode
        self.features = models.squeezenet1_1(pretrained=False).features
        self.classifier = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, outputs),
        )

    def forward(self, X):
        output = self.features(X)
        output = torch.mean(output.view(output.size(0), output.size(1), -1), dim=2)
        output = self.classifier(output)
        if self.mode == "classification":
            return F.log_softmax(output, dim=-1)
        elif self.mode == "regression":
            return F.sigmoid(output)
        else:
            raise Exception("Invalid mode: {}".format(self.mode))


class RasterNet3(nn.Module):
    def __init__(self, n_classes, mode="classification"):
        super(RasterNet3, self).__init__()
        self.mode = mode
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.05),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.05),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.05),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.05),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.05),
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.15),
            nn.Linear(256, n_classes),
        )

    def forward(self, X):
        output = self.conv(X)
        output = torch.mean(output.view(output.size(0), output.size(1), -1), dim=2)
        output = self.fc(output)

        if self.mode == "classification":
            return F.log_softmax(output, dim=-1)
        elif self.mode == "regression":
            return F.sigmoid(output)
        else:
            raise Exception("Invalid mode: {}".format(self.mode))


class RasterNet(nn.Module):
    def __init__(self, n_classes, mode="classification"):
        super(RasterNet, self).__init__()
        self.mode = mode
        self.conv = nn.Sequential(
            nn.Conv2d(5, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.05),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.05),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.05),
            nn.Conv2d(128, 256, 1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.05),
            nn.Conv2d(256, 512, 1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.05),
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.15),
            nn.Linear(256, n_classes),
        )

    def forward(self, X):
        output = self.conv(X)
        output = torch.mean(output.view(output.size(0), output.size(1), -1), dim=2)
        output = self.fc(output)

        if self.mode == "classification":
            return F.log_softmax(output, dim=-1)
        elif self.mode == "regression":
            return F.sigmoid(output)
        else:
            raise Exception("Invalid mode: {}".format(self.mode))
