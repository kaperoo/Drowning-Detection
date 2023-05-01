# PURPOSE: extract keypoints from dataset
import torch
import cv2
import os

kpts = torch.load('keypoints_norm17x3_no_misc.pt')

# discard the second part of the keypoints
kpts = kpts[:(len(kpts)//2), :, :]

print(kpts.shape)
# plot a histogram of how many keypoints had a confidence higher than 0.5
import matplotlib.pyplot as plt
import numpy as np

# get the confidence values
confidences = kpts[:, :, 2]

# initiate an array qith 17 elements, each one with a value of 0
confidences_count = np.zeros(17)

# iterate through the confidences array
for i in range(len(confidences)):
    for j in range(len(confidences[i])):
        # if the confidence is higher than 0.5, increment the corresponding element in confidences_count
        if confidences[i][j] > 0.5:
            confidences_count[j] += 1

# divide each element in confidences_count by the number of frames
confidences_count = [x/len(kpts) for x in confidences_count]


# plot the histogram
plt.bar(range(17), confidences_count)
plt.title("Confidence of keypoints")
plt.xlabel("Keypoint")
plt.ylabel("% keypoints with confidence > 0.5")

# set the xticks to the names of the keypoints
plt.xticks(range(17), ['nose', 'Reye', 'Leye', 'Rear', 'Lear', 'Lsho', 'Rsho', 'Lelb', 'Relb', 'Lwri', 'Rwri', 'Lhip', 'Rhip', 'Lkne', 'Rkne', 'Lank', 'Rank'])
# put the tics at 45 degrees
plt.xticks(rotation=45)

# make the figure size 40x20
plt.rcParams["figure.figsize"] = (40,20)

plt.show()
