#Purpose: Read the data from pt files and create a dataset for training
# Dataset of keypoint x,y,confidence and labels
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

folder = "C:\\Users\\User\\Desktop\\Code\\FYP\\code"

class KeypointDataset(Dataset):
    def __init__(self, annotations_file, keypoint_file, seq_length=10, transform=True, target_transform=True):
        self.img_labels = torch.load(os.path.join(folder, annotations_file))
        self.keypoint_file = torch.load(os.path.join(folder, keypoint_file))
        self.seq_length = seq_length
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels) - self.seq_length + 1

    def __getitem__(self, idx):
        start_idx = idx
        end_idx = idx + self.seq_length

        images_seq = self.keypoint_file[start_idx:end_idx]
        labels_seq = self.img_labels[start_idx:end_idx]

        if self.transform:
            images_seq = [torch.tensor(image).float() for image in images_seq]
            images_seq = torch.stack(images_seq, dim=0)

        if self.target_transform:
            labels_seq = [torch.tensor(label).float() for label in labels_seq]
            labels_seq = torch.stack(labels_seq, dim=0)

        return images_seq, labels_seq