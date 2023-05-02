#Purpose: Read the data from pt files and create a dataset for training
# Dataset of keypoint x,y,confidence and labels
import os
import torch
from torch.utils.data import Dataset

folder = "code"

# Dataset of keypoint x,y,confidence and labels
class KeypointDataset(Dataset):
    def __init__(self, annotations_file, keypoint_file, transform=True, target_transform=True):
        # define the path to the folder containing the data and labels
        self.img_labels = torch.load(os.path.join(folder, annotations_file))
        self.keypoint_file = torch.load(os.path.join(folder, keypoint_file))
        # define transform variables
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # get the image and label from the tensors
        image = self.keypoint_file[idx]
        label = self.img_labels[idx]
        # apply transforms so that the data is in the correct format
        if self.transform:
            image = torch.tensor(image).float()
        if self.target_transform:
            label = torch.tensor(label).float()
        return image, label