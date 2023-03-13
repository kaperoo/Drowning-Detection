import torch
from torch import load
from cnnmodel import CNNModel
import os

folder = "/home/kacperroemer/Code/FYP/keypoints_test"

# use model to predict
model = CNNModel().to('cuda:0')

with open('model_state.pt', 'rb') as f: 
    model.load_state_dict(load(f))
    
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
            outputs = torch.argmax(model(frames))
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