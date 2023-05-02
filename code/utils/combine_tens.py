# Purpose: prepare data for a faster CNNRNN training
import torch
import os

folder = "..\\datasets\\keypoints_norm"

# load and split all the data in the folder
for file in os.listdir(folder):

    if file.endswith(".pt"):
        # get the label from the file name
        label = file.split("_")[2]
        # ignore misc data
        if label == "misc":
            continue

        # load the data
        data = torch.load(os.path.join(folder, file))
        # split the data into tensors of 30 frames
        tensors = torch.split(data, 30, dim=0)

        # save sequences of 30 frames as separate tensors
        for i,t in enumerate(tensors):
            t = t.reshape(t.shape[0], 17, 3)
            # if the sequence is shorter than 30 frames, pad it with zeros
            if t.shape[0] < 30:
                t = torch.cat((t, torch.zeros(30-t.shape[0], 17, 3)), dim=0)

            print(t.shape)
            # save the data
            torch.save(t, "..\\keypoints_30\\" + file.split(".")[0] + "_" + str(i) + ".pt")
