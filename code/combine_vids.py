# Purpose: get tensors and labels from clips in ../train_frames
import torch
import torchvision
import os

folder = "../train_frames"

# get all the files in the folder
files = os.listdir(folder)

# create tensors and labels
tensors = torch.empty(0, 3, 120, 120)
labels = torch.empty(0, dtype=torch.long)

# load and concatenate all the data in the folder
for file in files:
    if file.endswith(".mp4"):
        video_tensor = torchvision.io.read_video(os.path.join(folder, file))[0].float()
        # tensor is [x, 120, 120, 3]
        # transpose to [x, 3, 120, 120]
        video_tensor = video_tensor.permute(0, 3, 1, 2)
        label = file.split("_")[2]
        if label == "drown":
            label = 0
        elif label == "swim":
            label = 1
        elif label == "misc":
            label = 2
        elif label == "idle":
            label = 3
        label_tensor = torch.tensor(int(label)).float()
        tensors = torch.cat((tensors, video_tensor), 0)
        labels = torch.cat((labels, torch.full((video_tensor.shape[0],), label, dtype=torch.long)), 0)

print(tensors.shape)
print(labels.shape)