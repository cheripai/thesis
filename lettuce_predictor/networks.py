import torch.nn as nn
from torchvision import models


class DenseNet(nn.Module):
    def __init__(self, outputs, p=0.1):
        super(DenseNet, self).__init__()
        self.model = models.densenet161(pretrained=True, drop_rate=p).cuda()
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(self.model.classifier.in_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
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

