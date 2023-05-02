#Purpose: Combine pt files into one file and create labels
import torch
import os

folder = "..\\datasets\\keypoints.pt"

# empty tensors for labels and keypoints
labels = torch.empty(0, dtype=torch.long)
# keypoints are different size tensors [x, 51]
keypoints = torch.empty(0, 51)

# load and concatenate all the data in the folder
for file in os.listdir(folder):
    if file.endswith(".pt"):
        # get the label from the file name
        label = file.split("_")[2]

        # ignore misc data
        if label == "misc":
            continue

        # load the data
        data = torch.load(os.path.join(folder, file))
        keypoints = torch.cat((keypoints, data), 0)

        # change labels to numbers
        if label == "drown":
            label = 0
        elif label == "swim":
            label = 1
        elif label == "idle":
            label = 2

        # create labels tensor
        labels = torch.cat((labels, torch.full((data.shape[0],), label, dtype=torch.long)), 0)

# reshape the keypoints tensor to be [num_frames, 17, 3]        
keypoints = keypoints.reshape(keypoints.shape[0], 17, 3)

# create a copy of the keypoints tensor
keypoints_mirror = keypoints.clone()

# augment the data with mirrored keypoints
for frame in keypoints_mirror:
    for kp in frame:
        kp[0] = 640 - kp[0]

# concatenate the original and mirrored keypoints
keypoints = torch.cat((keypoints, keypoints_mirror), 0)
# double the labels tensor
labels = torch.cat((labels, labels), 0)

print(keypoints.shape)

# save the data
torch.save(keypoints, "keypoints_norm17x3_no_misc.pt")
torch.save(labels, "labels_norm_no_misc.pt")
