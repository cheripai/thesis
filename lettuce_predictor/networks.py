import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DenseNet(nn.Module):
    def __init__(self, outputs, mode="classification", p=0.1):
        super(DenseNet, self).__init__()
        self.mode = mode
        self.model = models.densenet161(pretrained=True, drop_rate=p).cuda()
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(self.model.classifier.in_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, outputs),
        )

    def forward(self, X):
        output = self.model(X)
        if self.mode == "classification":
            return F.log_softmax(output, dim=-1)
        elif self.mode == "regression":
            return F.sigmoid(output)
        else:
            raise Exception("Invalid mode: {}".format(self.mode))


class ResNet(nn.Module):
    def __init__(self, outputs, mode="classification", p=0.1):
        super(ResNet, self).__init__()
        self.mode = mode
        self.model = models.resnet50(pretrained=True).cuda()
        self.model.fc = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(self.model.fc.in_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, outputs),
            nn.LogSoftmax(dim=-1),
        )
            
    def forward(self, X):
        output = self.model(X)
        if self.mode == "classification":
            return F.log_softmax(output, dim=-1)
        elif self.mode == "regression":
            return F.sigmoid(output)
        else:
            raise Exception("Invalid mode: {}".format(self.mode))


class VGG(nn.Module):
    def __init__(self, outputs, p=0.1):
        super(VGG, self).__init__()
        self.model = models.vgg16_bn(pretrained=True).features
        self.classifier = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            # nn.BatchNorm1d(4096),
            nn.Dropout(p=p),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            # nn.BatchNorm1d(1024),
            nn.Linear(1024, outputs),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, X):
        output = self.model(X)
        output = output.view(output.size(0), -1)
        return self.classifier(output)


class RasterNet(nn.Module):
    def __init__(self, n_classes, mode="classification"):
        super(RasterNet, self).__init__()
        self.mode = mode
        self.conv = nn.Sequential(
            nn.Conv2d(5, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout(p=0.11),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(p=0.11),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.Dropout(p=0.12),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.Dropout(p=0.15),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 1024),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, n_classes),
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

class RasterNetPlus(nn.Module):
    def __init__(self, n_classes, mode="classification"):
        super(RasterNetPlus, self).__init__()
        self.mode = mode
        self.conv = nn.Sequential(
            nn.Conv2d(5, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout(p=0.11),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(p=0.11),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.Dropout(p=0.12),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.Dropout(p=0.15),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.Dropout(p=0.25),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(p=0.25),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_classes),
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
