# PURPOSE: Dataset of videos and labels
# Videos from train_frames are 120x120
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import os

class VideoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.video_files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = os.path.join(self.root_dir, self.video_files[idx])
        video_tensor = torchvision.io.read_video(video_path)[0].float()
        label = self.video_files[idx].split('_')[2]
        if label == "drown":
            label = 0
        elif label == "swim":
            label = 1
        elif label == "misc":
            label = 2
        elif label == "idle":
            label = 3
        label_tensor = torch.tensor(int(label)).float()

        if self.transform:
            video_tensor = self.transform(video_tensor)

        return video_tensor, label_tensor