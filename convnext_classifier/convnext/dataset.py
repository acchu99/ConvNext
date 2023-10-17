import torch
from PIL import Image
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    def __init__(self, dataframe, transform=None, target_transform=None):
        self.df = dataframe
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        image = Image.open(self.df.iloc[idx, 1]).convert("RGB")
        label = self.df.iloc[idx, 2:].tolist()

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, torch.FloatTensor(label)
    