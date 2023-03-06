import torch
import os

folder = "/home/kacperroemer/Code/FYP/keypoints2.0"

labels = torch.empty(0, dtype=torch.long)
# keypoints are different size tensors [x, 51]
keypoints = torch.empty(0, 51)
# load and concatenate all the data in the folder
for file in os.listdir(folder):
    if file.endswith(".pt"):
        data = torch.load(os.path.join(folder, file))
        keypoints = torch.cat((keypoints, data), 0)
        # print(keypoints.shape)
        # print(file.split("_")[2]

        # write labels tensor
        label = file.split("_")[2]
        if label == "drown":
            label = 0
        elif label == "swim":
            label = 1
        elif label == "misc":
            label = 2
        elif label == "idle":
            label = 3

        labels = torch.cat((labels, torch.full((data.shape[0],), label, dtype=torch.long)), 0)
        # print(labels.shape)
        # print(keypoints.shape)
torch.save(keypoints, "keypoints.pt")
torch.save(labels, "labels.pt")
