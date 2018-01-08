import torch
from torchvision import transforms

# Path to store model
weights_path = 'data/weights.pth'

train_path = 'data/Lettuce_Equal/train'
valid_path = 'data/Lettuce_Equal/valid'

# Sliding window parameters
(win_width, win_height) = (40, 40)
step_size = 4

# Pyramid scale amount
scale_amt = 1.25

# Overlap threshold for NMS
overlap_thresh = 0.1

data_transforms = {
    "train":
    transforms.Compose([
        transforms.Resize(70),
        transforms.RandomCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ]),
    "valid":
    transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
    ]),
}

use_cuda = torch.cuda.is_available()
