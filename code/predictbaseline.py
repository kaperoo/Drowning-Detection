# PURPOSE: Predicts the class of a given video using the trained model
# and plots the confusion matrix
import torch
from torch import load
from baseline import NeuralNetwork
# from crnnmodel import CNNModel
import os
import matplotlib.pyplot as plt

folder = "C:\\Users\\User\\Desktop\\Code\\FYP\\keypoints_test_norm"

# use model to predict
model = NeuralNetwork().to('cuda:0')

with open('model_baseline.pt', 'rb') as f: 
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
        elif label == "idle":
            labelid = 2

        scores = [0,0,0]
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