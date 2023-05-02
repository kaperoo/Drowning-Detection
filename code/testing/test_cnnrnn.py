# PURPOSE: Predicts the class of a given video using CNNRNN model
# and plots the confusion matrix
import torch
from torch import load
import sys
sys.path.append("..\\models")
from cnnrnn import CNNRNN
import os
import matplotlib.pyplot as plt
import numpy as np

# folder containing the test data
folder = "..\\datasets\\keypoints_test_norm"

# load the model on the GPU
model = CNNRNN().to('cuda:0')

# load the model weights
with open('model_state_rnn.pt', 'rb') as f: 
    model.load_state_dict(load(f))

# loop over the test data and predict the class
scores_list = []
label_list = []
class_labels = ['drown', 'swim', 'idle']
seq_len = 30
for file in os.listdir(folder):
    if file.endswith(".pt"):
        
        # load the data
        test_data = torch.load(os.path.join(folder, file))

        # get the label from the file name
        label = file.split("_")[2]
        labelid = 4
        if label == "drown":
            labelid = 0
        elif label == "swim":
            labelid = 1
        elif label == "idle":
            labelid = 2

        # predict the class for each frame in the video
        scores = [0,0,0]
        frame_buffer = []
        for i in range(0,len(test_data)):    

            # append a frame to the buffer
            frame_buffer.append(test_data[i].unsqueeze(-1).to('cuda:0'))
            
            # remove the first frame if the buffer is full (max length = seq_len)
            if len(frame_buffer) > seq_len:
                frame_buffer.pop(0)

            # minimum number of frames in the buffer to make a prediction
            if len(frame_buffer) > 5:

                # prepare the data for the model
                keypoints_tensor = torch.stack(frame_buffer, dim=0).unsqueeze(1).to('cuda:0')
                keypoints_tensor = keypoints_tensor.reshape(1, len(frame_buffer), 17, 3, 1)

                # predict the class
                with torch.no_grad():
                    output = model(keypoints_tensor)

                # Get the label of the most frequent class in the sequence
                outputs = np.argmax(output.cpu().numpy())
                scores_list.append(outputs)
                label_list.append(labelid)
                scores[outputs] += 1

        # calculate the accuracy for the whole video
        accuracy = scores[labelid] / sum(scores)

        # find index of max score
        max_score = max(scores)
        max_index = scores.index(max_score)

        # check if the prediction of the whole video is correct
        if max_index == labelid:
            predicted = "correct"
        else:
            predicted = "false"

        # print the results
        print(f"{label:7} {predicted:8} {accuracy:.2f} {scores}")

# calculate accuracy of the model
correct = 0
for i in range(len(scores_list)):
    if scores_list[i] == label_list[i]:
        correct += 1

print(f"Accuracy: {correct/len(scores_list):.2f}")

# calculate the precision, recall and f1 score
from sklearn.metrics import precision_recall_fscore_support
precision, recall, f1, _ = precision_recall_fscore_support(label_list, scores_list, average='macro')
print(f"Precision: {precision:.2f} Recall: {recall:.2f} F1: {f1:.2f}")

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
