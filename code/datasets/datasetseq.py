#Purpose: Read the data from pt files and create a dataset for training
# Dataset of keypoint x,y,confidence and labels
import os
import torch
from torch.utils.data import Dataset

folder = "code"

# Dataset of keypoint x,y,confidence and labels
class KeypointDataset(Dataset):
    def __init__(self, annotations_file, keypoint_file, seq_length=10, transform=True, target_transform=True):
        # define the path to the folder containing the data and labels
        self.img_labels = torch.load(os.path.join(folder, annotations_file))
        self.keypoint_file = torch.load(os.path.join(folder, keypoint_file))
        # define transform variables and sequence length
        self.seq_length = seq_length
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels) - self.seq_length + 1

    def __getitem__(self, idx):
        # initiate the start and end index of the sequence
        start_idx = idx
        end_idx = idx + self.seq_length

        # get the image and label sequence from the tensors
        images_seq = self.keypoint_file[start_idx:end_idx]
        labels_seq = self.img_labels[start_idx:end_idx]

        # apply transforms so that the data is in the correct format
        if self.transform:
            images_seq = [torch.tensor(image).float() for image in images_seq]
            images_seq = torch.stack(images_seq, dim=0)

        if self.target_transform:
            labels_seq = [torch.tensor(label).float() for label in labels_seq]
            labels_seq = torch.stack(labels_seq, dim=0)

        return images_seq, labels_seq