import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from imageloader import PairedImageLoader
from models.fcdensenet import FCDenseNet
from scipy.misc import imsave
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms


epochs = 100

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
])

if __name__ == "__main__":
    train_set = PairedImageLoader("data/x_train", "data/y_train", transform=transform)
    valid_set = PairedImageLoader("data/x_valid", "data/y_valid", transform=transforms.ToTensor())
    trainloader  = DataLoader(train_set, batch_size=3, shuffle=True, num_workers=2, pin_memory=True)
    validloader  = DataLoader(valid_set, batch_size=3, shuffle=False, num_workers=2, pin_memory=True)

    model = FCDenseNet(4, 12, drop_rate=0.0, n_classes=1).cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=5)
    criterion = nn.MSELoss()

    for i in range(epochs):
        print("Epoch", i)
        total_loss = 0
        for j, (x, y) in enumerate(trainloader):
            x, y = Variable(x.cuda()), Variable(y.cuda())
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.data[0]
        scheduler.step(total_loss)
        print("Train loss:", total_loss / j)

        total_loss = 0
        for j, (x, y) in enumerate(validloader):
            x, y = Variable(x.cuda()), Variable(y.cuda())
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.data[0]

            for k in range(outputs.size(0)):
                sample = outputs[k].data.cpu().squeeze().numpy() * 255
                imsave("results/sample{}_{}.jpg".format(j, k), sample)

        print("Valid loss:", total_loss / j)

        torch.save(model.state_dict(), "data/weights.pth")
