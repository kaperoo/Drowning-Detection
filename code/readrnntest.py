#Purpose: Read the data from pt files and create a dataset for training
# Dataset of keypoint x,y,confidence and labels
import os
import pandas as pd
import torch
from torch.utils.data import Dataset

class KeypointDataset(Dataset):
    def __init__(self, keypoint_file, transform=True, target_transform=True):
        
        self.keypoint_file = keypoint_file
        self.transform = transform
        self.target_transform = target_transform
        self.tensors = os.listdir(keypoint_file)

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        t_path = os.path.join(self.keypoint_file, self.tensors[idx])
        tensor = torch.load(t_path)

        label = self.tensors[idx].split('_')[2]
        if label == "drown":
            label = 0
        elif label == "swim":
            label = 1
        elif label == "idle":
            label = 2

        labels_seq = [label] * len(tensor)

        if self.transform:
            tensor = [torch.tensor(t).float() for t in tensor]
            tensor = torch.stack(tensor, dim=0)

        if self.target_transform:
            labels_seq = [torch.tensor(label).float() for label in labels_seq]
            labels_seq = torch.stack(labels_seq, dim=0)

        return tensor, labels_seq