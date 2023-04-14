#Purpose: Combine pt files into one file and create labels
import torch
import os

folder = "/home/kacperroemer/Code/FYP/keypoints_norm"
# folder = "/home/kacperroemer/Code/FYP/keypoints_test"


labels = torch.empty(0, dtype=torch.long)
# keypoints are different size tensors [x, 51]
keypoints = torch.empty(0, 51)
# load and concatenate all the data in the folder
for file in os.listdir(folder):
    if file.endswith(".pt"):
        label = file.split("_")[2]
        if label == "misc":
            continue
        data = torch.load(os.path.join(folder, file))
        keypoints = torch.cat((keypoints, data), 0)
        # print(keypoints.shape)
        # print(file.split("_")[2]

        # write labels tensor
        if label == "drown":
            label = 0
        elif label == "swim":
            label = 1
        # elif label == "misc":
            # label = 2
        elif label == "idle":
            label = 2

        labels = torch.cat((labels, torch.full((data.shape[0],), label, dtype=torch.long)), 0)
        
# torch.save(keypoints, "keypoints_test.pt")
keypoints = keypoints.reshape(keypoints.shape[0], 17, 3)

# create a copy of the keypoints tensor
keypoints_mirror = keypoints.clone()

# augment the data with mirrored keypoints
for frame in keypoints_mirror:
    for kp in frame:
        kp[0] = 640 - kp[0]

keypoints = torch.cat((keypoints, keypoints_mirror), 0)
labels = torch.cat((labels, labels), 0)

print(keypoints.shape)


torch.save(keypoints, "keypoints_norm17x3_no_misc.pt")
torch.save(labels, "labels_norm_no_misc.pt")
