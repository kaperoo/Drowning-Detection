#Purpose: Read the data from pt files and create a dataset for training
# Dataset of keypoint x,y,confidence and labels
import os
import torch
from torch.utils.data import Dataset

# Subclass of Dataset to load the data
class KeypointDataset(Dataset):
    def __init__(self, keypoint_file, transform=True, target_transform=True):
        
        # define the path to the folder containing the data and labels
        self.keypoint_file = keypoint_file
        self.transform = transform
        # define transform variables
        self.target_transform = target_transform
        # get the list of sequence tensors
        self.tensors = os.listdir(keypoint_file)

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        # get the image and label from the tensors
        t_path = os.path.join(self.keypoint_file, self.tensors[idx])
        tensor = torch.load(t_path)

        # change the label to a number
        label = self.tensors[idx].split('_')[2]
        if label == "drown":
            label = 0
        elif label == "swim":
            label = 1
        elif label == "idle":
            label = 2

        # create a tensor of labels for each frame in the sequence
        labels_seq = [label] * len(tensor)

        # apply transforms so that the data is in the correct format
        if self.transform:
            tensor = [torch.tensor(t).float() for t in tensor]
            tensor = torch.stack(tensor, dim=0)

        if self.target_transform:
            labels_seq = [torch.tensor(label).float() for label in labels_seq]
            labels_seq = torch.stack(labels_seq, dim=0)

        return tensor, labels_seq