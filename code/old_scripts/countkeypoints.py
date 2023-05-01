# PURPOSE: extract keypoints from dataset
import torch

import cv2
import os

folder = [
          '..\\dataset\\train\\tr_underwater\\tr_u_drown', 
          '..\\dataset\\train\\tr_underwater\\tr_u_swim', 
        #   '../dataset/train/tr_underwater/tr_u_misc', 
          '..\\dataset\\train\\tr_underwater\\tr_u_idle',
          '..\\dataset\\train\\tr_overhead\\tr_o_drown',
          '..\\dataset\\train\\tr_overhead\\tr_o_swim',
        #   '../dataset/train/tr_overhead/tr_o_misc',
          '..\\dataset\\train\\tr_overhead\\tr_o_idle'
        #   '../dataset/test/te_underwater',
        #   '../dataset/test/te_overhead'
        ]

# count the number of total frames
frames = [0,0,0]
keypints = [0,0,0]

for directory in folder:
    for video_filename in os.listdir(os.path.join(directory)):
        label = video_filename.split("_")[2]
        if label == "drown":
            label = 0
        elif label == "swim":
            label = 1
        elif label == "idle":
            label = 2
        video = cv2.VideoCapture(os.path.join(directory, video_filename))
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frames[label] += num_frames
        video.release()
print(frames)
print(sum(frames))

lbls = torch.load('labels_norm_no_misc.pt')
for l in lbls:
    label = l.item()
    if label == "drown":
        label = 0
    elif label == "swim":
        label = 1
    elif label == "idle":
        label = 2
    keypints[label] += 1

# divide each element in keypints by 2
keypints = [int(x/2) for x in keypints]

print(keypints)
print(sum(keypints))

# plot barchart comparing frames and keypoints
import matplotlib.pyplot as plt
import numpy as np

labels = ['drown', 'swim', 'idle']
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, frames, width, label='frames')
rects2 = ax.bar(x + width/2, keypints, width, label='keypoints')

# put exact values on top of bars
for rect in rects1:
    height = rect.get_height()
    ax.annotate('{}'.format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, -20),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')
for rect in rects2:
    height = rect.get_height()
    ax.annotate('{}'.format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, -20),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Number of frames')
ax.set_title('Number of frames and keypoints')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()

plt.savefig('pe.png')

plt.show()