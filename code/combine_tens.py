import torch
import os

folder = "C:\\Users\\User\\Desktop\\Code\\FYP\\keypoints_norm"

for file in os.listdir(folder):
    if file.endswith(".pt"):
        label = file.split("_")[2]
        if label == "misc":
            continue
        data = torch.load(os.path.join(folder, file))
        tensors = torch.split(data, 30, dim=0)

        for i,t in enumerate(tensors):
            t = t.reshape(t.shape[0], 17, 3)
            if t.shape[0] < 30:
                # append zeros to end of tensor
                t = torch.cat((t, torch.zeros(30-t.shape[0], 17, 3)), dim=0)
            print(t.shape)
            torch.save(t, "C:\\Users\\User\\Desktop\\Code\\FYP\\keypoints_30\\" + file.split(".")[0] + "_" + str(i) + ".pt")
