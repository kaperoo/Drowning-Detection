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
        # read image as a tensor
        video_tensor = torchvision.io.read_image(video_path).float()
        # video_tensor = torchvision.io.read_video(video_path)[0].float()
        label = self.video_files[idx].split('_')[2]
        if label == "drown":
            label = 0
        elif label == "swim":
            label = 1
        elif label == "idle":
            label = 2
        label_tensor = torch.tensor(int(label)).float()

        if self.transform:
            video_tensor = self.transform(video_tensor)

        return video_tensor, label_tensor

# class VideoDataset(Dataset):
#     def __init__(self, root_dir, seq_length, transform=None):
#         self.root_dir = root_dir
#         self.seq_length = seq_length
#         self.transform = transform
#         self.video_files = os.listdir(self.root_dir)

#     def __len__(self):
#         return len(self.video_files)

#     def __getitem__(self, idx):
#         video_path = os.path.join(self.root_dir, self.video_files[idx])
#         video_tensor = torchvision.io.read_video(video_path)[0].float()
        
#         # Create sequences of the specified length
#         num_frames = video_tensor.shape[0]
#         start_idx = torch.randint(0, max(1, num_frames - self.seq_length + 1), (1,)).item()
#         video_tensor = video_tensor[start_idx:start_idx + self.seq_length, :, :, :]

#         label = self.video_files[idx].split('_')[2]
#         if label == "drown":
#             label = 0
#         elif label == "swim":
#             label = 1
#         elif label == "idle":
#             label = 2
#         label_tensor = torch.tensor(int(label)).float()

#         if self.transform:
#             video_tensor = self.transform(video_tensor)

#         return video_tensor, label_tensor

# class VideoDataset(Dataset):
#     def __init__(self, root_dir, seq_length, transform=None):
#         self.root_dir = root_dir
#         self.seq_length = seq_length
#         self.transform = transform
#         self.video_files = os.listdir(self.root_dir)
#         self.sequences, self.labels = self._extract_sequences()

#     def __len__(self):
#         return len(self.sequences)

#     def __getitem__(self, idx):
#         video_tensor = self.sequences[idx]
#         label_tensor = self.labels[idx]

#         if self.transform:
#             video_tensor = self.transform(video_tensor)

#         return video_tensor, label_tensor

#     def _extract_sequences(self):
#         sequences = []
#         labels = []

#         for video_file in self.video_files:
#             video_path = os.path.join(self.root_dir, video_file)
#             video_tensor = torchvision.io.read_video(video_path)[0].float()

#             num_frames = video_tensor.shape[0]
#             for start_idx in range(0, num_frames - self.seq_length + 1, self.seq_length):
#                 seq = video_tensor[start_idx:start_idx + self.seq_length, :, :, :]
#                 sequences.append(seq)

#                 label = video_file.split('_')[2]
#                 if label == "drown":
#                     label = 0
#                 elif label == "swim":
#                     label = 1
#                 elif label == "idle":
#                     label = 2
#                 label_tensor = torch.tensor(int(label)).float()
#                 labels.append(label_tensor)

#         return sequences, labels