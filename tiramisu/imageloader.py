import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class PairedImageLoader(Dataset):
    def __init__(self, x_dir, y_dir, transform=None):
        self.transform = transform
        self.x_paths = sorted([os.path.join(x_dir, f) for f in os.listdir(x_dir)])
        self.y_paths = sorted([os.path.join(y_dir, f) for f in os.listdir(y_dir)])

        if len(self.x_paths) != len(self.y_paths):
            raise Exception("Number of files in {} and {} do not match!".format(x_dir, y_dir))

    def __getitem__(self, index):
        x = Image.open(self.x_paths[index]).convert("RGB")
        y = Image.open(self.y_paths[index]).convert("L")

        if self.transform is not None:
            x = self.transform(x)
            y = self.transform(y)

        return x, y

    def __len__(self):
        return len(self.x_paths)
