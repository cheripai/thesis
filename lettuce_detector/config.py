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
overlap_thresh = 0.2

# Model Configuration
use_cuda = torch.cuda.is_available()
batch_size = 32
img_size = 64

data_transforms = {
    "train":
    transforms.Compose([
        transforms.Resize(int(img_size*1.1)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ]),
    "valid":
    transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ]),
}

