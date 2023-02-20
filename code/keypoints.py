import torch
from torchvision import transforms
from os.path import dirname, join

import sys
sys.path.append(join(dirname(__file__), "../yolov7"))

from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts

import matplotlib.pyplot as plt
import cv2
import numpy as np
import os


# Load YOLOv7 model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)

def load_model():
    model = torch.load(join(dirname(__file__), "../yolov7/yolov7-w6-pose.pt"), map_location=device)['model']
    # Put in inference mode
    model.float().eval()

    if torch.cuda.is_available():
        print("Using GPU")
        # half() turns predictions into float16 tensors
        # which significantly lowers inference time
        model.half().to(device)
    return model

model = load_model()

# inference function
def run_inference(image):
    # Resize and pad image
    image = letterbox(image, 960, stride=64, auto=True)[0] # shape: (567, 960, 3)
    # Apply transforms
    image = transforms.ToTensor()(image) # torch.Size([3, 567, 960])
    if torch.cuda.is_available():
        image = image.half().to(device)
    # Turn image into batch
    image = image.unsqueeze(0) # torch.Size([1, 3, 567, 960])
    with torch.no_grad():
        output, _ = model(image)
    return output


# folder = '../dataset/train/tr_underwater/tr_u_drown'
# folder = '../dataset/train/tr_underwater/tr_u_swim'
# folder = '../dataset/train/tr_overhead'
folder = ['../dataset/train/tr_underwater/tr_u_misc', '../dataset/train/tr_underwater/tr_u_idle']

# Loop over videos in the folder
# for directory in os.listdir(folder):
for directory in folder:
    # for video_filename in os.listdir(os.path.join(folder, directory)):
    for video_filename in os.listdir(os.path.join(directory)):
        if not video_filename.endswith(".mp4"):
            continue
        print(video_filename)

        # Load video
        # cap = cv2.VideoCapture(os.path.join(folder, directory, video_filename))
        cap = cv2.VideoCapture(os.path.join(directory, video_filename))

        # Create lists to store keypoints and class labels
        keypoints_list = []

        # Loop over video frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLOv7 on the frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output = run_inference(frame)

            # Extract keypoints from the output
            keypoints = []
            keypoints = non_max_suppression_kpt(output, 
                                        0.25, # Confidence Threshold
                                        0.65, # IoU Threshold
                                        nc=model.yaml['nc'], # Number of Classes
                                        nkpt=model.yaml['nkpt'], # Number of Keypoints
                                        kpt_label=True)
            with torch.no_grad():
                keypoints = output_to_keypoint(keypoints)



            # Convert keypoints to PyTorch tensor
            keypoints = torch.tensor(keypoints).float()

            # if more than one person is detected, take the one with the highest confidence
            if keypoints.shape[0] > 1:
                # keep only the keypoints of the person with the highest confidence
                keypoints = keypoints[torch.argmax(keypoints[:, 6]), 7:]
            elif keypoints.shape[0] == 1:
                keypoints = keypoints[0, 7:]
            else:
                continue

            # Save keypoints
            keypoints_list.append(keypoints)

        # Release video capture
        cap.release()

        # Convert list to PyTorch tensor
        keypoints_tensor = torch.stack(keypoints_list)

        # Save keypoints to file
        torch.save(keypoints_tensor, os.path.join("../keypoints", video_filename + ".pt"))
