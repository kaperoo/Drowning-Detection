#Purpose: Read the data from pt files and create a dataset for training
# Dataset of keypoint x,y,confidence and labels
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

folder = "/home/kacperroemer/Code/FYP/code"

class KeypointDataset(Dataset):
    def __init__(self, annotations_file, keypoint_file, transform=True, target_transform=True):
        self.img_labels = torch.load(os.path.join(folder, annotations_file))
        self.keypoint_file = torch.load(os.path.join(folder, keypoint_file))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.keypoint_file[idx]
        label = self.img_labels[idx]
        if self.transform:
            # image = self.transform(image)
            image = torch.tensor(image).float()
        if self.target_transform:
            # label = self.target_transform(label)
            label = torch.tensor(label).float()
        return image, label