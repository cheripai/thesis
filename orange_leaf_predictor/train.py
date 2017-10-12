import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
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
        with open(label_file) as f:
            labels = [float(line.strip()) for line in f.readlines()]
        img_paths, labels = shuffle(img_paths, labels)
        split = int(len(img_paths) * p)
        return LeafDataset(img_paths[split:], labels[split:]), LeafDataset(img_paths[:split], labels[:split])


class DenseNet(nn.Module):
    def __init__(self, features=128):
        super(DenseNet, self).__init__()
        self.model = models.densenet121(pretrained=True).cuda()
        self.model.classifier = nn.Linear(self.model.classifier.in_features, features)
        self.dense = nn.Linear(features, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, X):
        output = self.relu(self.model(X))
        return self.tanh(self.dense(output))

        

if __name__ == "__main__":
    leaf_train, leaf_valid = get_train_test("data/img", "data/ndvi.txt")
    train_loader = DataLoader(leaf_train, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(leaf_valid, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    model = DenseNet().cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 5
    for i in range(epochs):
        epoch_loss = 0
        for X, y in train_loader:
            X, y = Variable(X.cuda()), Variable(y.type(torch.FloatTensor).cuda())
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.data[0]
        print("Epoch {} Train Loss {}".format(i, epoch_loss))

        valid_loss = 0
        for X, y in valid_loader:
            X, y = Variable(X.cuda()), Variable(y.type(torch.FloatTensor).cuda())
            outputs = model(X)
            loss = criterion(outputs, y)
            valid_loss += loss.data[0]
        print("Epoch {} Validation Loss {}".format(i, valid_loss))
