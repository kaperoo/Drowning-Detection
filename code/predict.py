# PURPOSE: Predicts the class of a given video using the trained model
# and plots the confusion matrix
import torch
from torch import load
from cnnmodel import CNNModel
# from crnnmodel import CNNModel
import os
import matplotlib.pyplot as plt
import numpy as np

folder = "/home/kacperroemer/Code/FYP/keypoints_test_norm"

# use model to predict
model = CNNModel().to('cuda:0')

with open('model_state.pt', 'rb') as f: 
    model.load_state_dict(load(f))
    
scores_list = []
label_list = []

for file in os.listdir(folder):
    if file.endswith(".pt"):
        
        test_data = torch.load(os.path.join(folder, file))

        label = file.split("_")[2]
        labelid = 4
        if label == "drown":
            labelid = 0
        elif label == "swim":
            labelid = 1
        elif label == "misc":
            labelid = 2
        elif label == "idle":
            labelid = 3

        scores = [0,0,0,0]
        for i in range(len(test_data)):    
            frames = test_data[i].unsqueeze(0).to('cuda:0')
            # print(frames.shape)
            # frames = frames.reshape(1, 17, 1, 3)
            outputs = torch.argmax(model(frames))
            scores_list.append(outputs.cpu().detach().numpy())
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
df_cm = pd.DataFrame(cm, index = [i for i in ["drown", "swim", "misc", "idle"]],
                    columns = [i for i in ["drown", "swim", "misc", "idle"]])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True) #, fmt='g'

plt.show()