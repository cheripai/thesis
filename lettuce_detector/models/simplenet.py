import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class SimpleNet(nn.Module):
    def __init__(self, n_classes):
        super(SimpleNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 7, stride=2, padding=3),
            nn.Dropout(p=0.1),
            nn.ELU(inplace=True),
            nn.Conv2d(16, 32, 5, stride=2, padding=2),
            nn.Dropout(p=0.1),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.Dropout(p=0.2),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 128, 5, stride=2, padding=2),
            nn.Dropout(p=0.2),
            nn.ELU(inplace=True),
        )

        self.fc = nn.Sequential(nn.Linear(128, n_classes), nn.LogSoftmax(dim=-1))

    def forward(self, X):
        output = self.conv(X)
        output = torch.mean(output.view(output.size(0), output.size(1), -1), dim=2)
        return self.fc(output)
