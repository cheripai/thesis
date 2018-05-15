import config as c
import numpy as np
import torch
import torch.nn as nn
from imageloader import PairedImageLoader
from models.fcdensenet import FCDenseNet
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

X_PATH = "data/x_valid"
Y_PATH = "data/y_valid"

if __name__ == "__main__":
    eval_set = PairedImageLoader(X_PATH, Y_PATH, transform=transforms.ToTensor())
    loader = DataLoader(eval_set, batch_size=c.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = FCDenseNet([4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4], 16, drop_rate=0.0, n_classes=1).cuda()
    model.load_state_dict(torch.load(c.WEIGHTS_PATH))
    model.train(False)
    criterion = nn.L2Loss()

    total_loss = 0
    for j, (x, y) in enumerate(loader):
        x, y = Variable(x.cuda()), Variable(y.cuda())
        outputs = model(x)
        loss = criterion(outputs * 255, y * 255)
        total_loss += loss.data[0]

    print("MSE per pixel:", total_loss / (j + 1))
