# PURPOSE: Predicts the class of a given video using the trained model
# and plots the confusion matrix
import torch
from torch import load
from videomodel2d import VideoCNN
import torchvision
import os
import matplotlib.pyplot as plt
import numpy as np

folder = "C:\\Users\\User\\Desktop\\Code\\FYP\\test_frames"

# use model to predict
model = VideoCNN().to('cuda:0')

with open('model_state_frames2d.pt', 'rb') as f: 
    model.load_state_dict(load(f))
    
scores_list = []
label_list = []

class_labels = ['drown', 'swim', 'idle']

seq_len = 10

for file in os.listdir(folder):
    if file.endswith(".mp4"):
        
        test_data = torchvision.io.read_video(os.path.join(folder, file))[0].float()

        label = file.split("_")[2]
        labelid = 4
        if label == "drown":
            labelid = 0
        elif label == "swim":
            labelid = 1
        elif label == "idle":
            labelid = 2

        scores = [0,0,0]
        for i in range(0,len(test_data)):    
            frame = test_data[i].to('cuda:0')
            # Forward pass through the model
            with torch.no_grad():
                output = model(frame)

            outputs = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            scores_list.append(outputs)
            label_list.append(labelid)
            scores[outputs] += 1


        accuracy = scores[labelid] / sum(scores)

        # find index of max score
        max_score = max(scores)
        max_index = scores.index(max_score)

        if max_index == labelid:
            predicted = "correct"
        else:
            predicted = "false"

        print(f"{label:7} {predicted:8} {accuracy:.2f} {scores}")

# plot the confusion matrix of the predictions
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

cm = confusion_matrix(label_list, scores_list, normalize='true') #normalize='true'
df_cm = pd.DataFrame(cm, index = [i for i in ["drown", "swim", "idle"]],
                    columns = [i for i in ["drown", "swim", "idle"]])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True) #, fmt='g'

plt.show()