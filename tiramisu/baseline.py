import config as c
import numpy as np
from imageloader import PairedImageLoader
from sklearn.linear_model import SGDRegressor
from torch.utils.data import DataLoader
from torchvision import transforms

model = SGDRegressor(fit_intercept=False)

train_set = PairedImageLoader("data/x_train", "data/y_train", transform=transforms.ToTensor())
valid_set = PairedImageLoader("data/x_valid", "data/y_valid", transform=transforms.ToTensor())
trainloader  = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
validloader  = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

def rmse(y, yhat):
    return np.sqrt(np.mean((y - yhat)**2))

if __name__ == "__main__":
    total_loss = 0
    for j, (x, y) in enumerate(trainloader):
        x = x.squeeze().view(-1, c.IMG_SIZE**2).numpy().T
        y = y.squeeze().view(c.IMG_SIZE**2).numpy()
        model.partial_fit(x, y)

    total_loss = 0
    for j, (x, y) in enumerate(validloader):
        x = x.squeeze().view(-1, c.IMG_SIZE**2).numpy().T
        y = y.squeeze().view(c.IMG_SIZE**2).numpy()
        predictions = model.predict(x)
        total_loss += rmse(y, predictions)

    print("Test loss:", total_loss / j)
