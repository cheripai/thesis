import config as cfg
import torch
import torch.nn as nn
import torch.optim as optim
from models.simplenet import SimpleNet
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from utils import num_correct

if __name__ == "__main__":
    batch_size = 32
    lr = 0.001
    train_loader = DataLoader(
        ImageFolder(cfg.train_path, cfg.data_transforms["train"]),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True)
    valid_loader = DataLoader(
        ImageFolder(cfg.valid_path, cfg.data_transforms["valid"]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True)

    model = SimpleNet(2).cuda()
    criterion = nn.NLLLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Learning rate: {}".format(lr))

    epochs = 5
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
            train_acc += num_correct(outputs, y)
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
            valid_acc += num_correct(outputs, y)
            count += X.size(0)
        valid_acc /= count
        print("Valid Loss {} Valid Accuracy {}".format(round(valid_loss, 4), round(valid_acc, 4)))

    torch.save(model.state_dict(), cfg.weights_path)
