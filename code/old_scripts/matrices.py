import os

labels = []
folder = "C:\\Users\\User\\Desktop\\Code\\FYP\\keypoints_test_norm"
for file in os.listdir(folder):
    label = file.split("_")
    labels.append(str(label[1] + "_" + label[2] + "_" + label[3].split(".")[0]))

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Define the labels and models
label_list = [0,0,0,0,2,2,1,1,1,0,0,0,0,0,0,2,2,2,1,1,1]
models = ['Baseline', 'CNN-SINGLE', 'CNN-SEQ', 'CNN-RNN', '3D-CNN']

# Define the predicted labels
baseline = [0,1,1,0,2,2,1,1,1,0,1,0,0,0,0,2,2,0,1,1,1]
b = [0.94,0.29,0.15,0.58,0.64,0.94,0.67,0.97,0.72,0.69,0.10,0.66,0.79,0.73,0.67,0.80,0.62,0.15,0.70,1.00,0.95]
cnn = [0,1,1,0,2,2,1,1,1,0,1,0,0,0,0,2,2,0,1,1,1]
c = [0.98 ,0.34 ,0.32 ,0.73 ,0.72 ,1.00 ,0.64 ,0.99 ,0.75 ,0.68 ,0.09 ,0.64 ,0.67 ,0.65 ,0.68 ,0.53 ,0.67 ,0.03 ,0.72 ,1.00 ,0.91]
cnnseq = [0,1,1,0,2,2,1,1,1,0,1,0,0,0,0,2,2,0,1,1,1]
s = [1.00 ,0.29 ,0.16 ,0.93 ,0.83 ,1.00 ,0.73 ,1.00 ,0.91 ,0.77 ,0.00 ,0.90 ,1.00 ,0.98 ,0.75 ,0.60 ,0.70 ,0.00 ,0.73 ,1.00 ,0.69 ]
cnnrnn = [0,0,1,0,2,2,1,1,1,0,1,0,0,0,0,2,2,0,0,1,1]
r = [0.99 ,0.62 ,0.26 ,0.75 ,0.92 ,1.00 ,0.66 ,0.79 ,0.65 ,0.84 ,0.16 ,0.84 ,1.00 ,0.97 ,0.86 ,0.96 ,0.89 ,0.30 ,0.10 ,0.92 ,0.99] 
threedcnn = [0,1,1,0,2,2,1,1,1,0,1,0,0,0,0,2,2,0,1,1,1]
t =[0.89 ,0.28 ,0.12 ,0.44 ,0.63 ,0.97 ,0.40 ,0.79 ,0.68 ,0.53 ,0.17 ,0.51 ,0.72 ,0.72 ,0.69 ,0.76 ,0.61 ,0.31 ,0.57 ,0.88 ,0.63]
data = [baseline, cnn, cnnseq, cnnrnn, threedcnn]

actual = [b,c,s,r,t]

for i in data:
    for j in range(0, len(i)):
        if i[j] == label_list[j]:
            i[j] = 1
        else:
            i[j] = 0

# Define the colors for correct and incorrect predictions
cmap = sns.color_palette(['red', 'green'])

# Create the plot
sns.set(style='white')
fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(data, cmap=cmap, annot=True, fmt='.0f', cbar=False, xticklabels=labels, yticklabels=models, square=True, annot_kws={"color": "white", "fontsize": 12}, cbar_kws={"ticks":[0, 1]}, ax=ax, linewidths=1, linecolor='black')

# put the xticks at an angle and move slightly to the left
plt.xticks(rotation=45, ha='right', rotation_mode='anchor')

for txt in ax.texts:
    # set text to ""
    txt.set_text("")
# change the text in the grid to data from actual list
for i in range(0, len(models)):
    for j in range(0, len(labels)):
        text = ax.text(j+0.5, i+0.5, actual[i][j], ha="center", va="center", color="white", fontsize=12)


# Set the plot title and axis labels
plt.title('Classification results on entire videos', fontsize=16)

# Show the plot
plt.show()