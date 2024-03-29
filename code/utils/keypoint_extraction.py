# PURPOSE: extract keypoints from dataset with YOLOv7
import torch
from torchvision import transforms
from os.path import dirname, join

import sys
sys.path.append(join(dirname(__file__), "..\\..\\yolov7"))

from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint

import cv2
import os
import numpy as np


# find the device available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define a function to load the model
def load_model():

    # load the model
    model = torch.load(join(dirname(__file__), "../yolov7/yolov7-w6-pose.pt"), map_location=device)['model']
    # Put in inference mode
    model.float().eval()

    if torch.cuda.is_available():
        print("Using GPU")
        # half() turns predictions into float16 tensors
        # which significantly lowers inference time
        model.half().to(device)
    return model

# load the model
model = load_model()

# inference function
def run_inference(image):
    # Resize and pad image for faster inference
    image = letterbox(image, new_shape = (640), stride=64, auto=True)[0]

    # Apply transforms
    image = transforms.ToTensor()(image) 
    if torch.cuda.is_available():
        image = image.half().to(device)

    # Turn image into batch
    image = image.unsqueeze(0)

    # Run inference
    with torch.no_grad():
        output, _ = model(image)
    return output

# define the folder to be processed
folder = [
          '../dataset/train/tr_underwater/tr_u_drown', 
          '../dataset/train/tr_underwater/tr_u_swim', 
          '../dataset/train/tr_underwater/tr_u_misc', 
          '../dataset/train/tr_underwater/tr_u_idle',
          '../dataset/train/tr_overhead/tr_o_drown',
          '../dataset/train/tr_overhead/tr_o_swim',
          '../dataset/train/tr_overhead/tr_o_misc',
          '../dataset/train/tr_overhead/tr_o_idle'
        #   '../dataset/test/te_underwater',
        #   '../dataset/test/te_overhead'
        ]

# Loop over videos in the folders
for directory in folder:
    for video_filename in os.listdir(os.path.join(directory)):
        if not video_filename.endswith(".mp4"):
            continue

        # Create lists to store keypoints
        keypoints_list = []

        # Load video
        cap = cv2.VideoCapture(os.path.join(directory, video_filename))
    
        # Loop over video frames
        while True:        
            # print progress
            os.system('cls' if os.name == 'nt' else 'clear')
            print("Processing video:", video_filename)
            print("Frame:", cap.get(cv2.CAP_PROP_POS_FRAMES), "/", cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print("Video:", os.listdir(directory).index(video_filename) + 1, "/", len(os.listdir(directory)))

            # Read frame
            ret, frame = cap.read()
            if not ret:
                break

            # apply color conversion and run inference on frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output = run_inference(frame)

            # Extract keypoints from the output by applying non-max suppression
            keypoints = []
            keypoints = non_max_suppression_kpt(output, 
                                        0.25, # Confidence Threshold
                                        0.65, # IoU Threshold
                                        nc=model.yaml['nc'], # Number of Classes
                                        nkpt=model.yaml['nkpt'], # Number of Keypoints
                                        kpt_label=True)

            # Convert output to keypoints
            with torch.no_grad():
                keypoints = output_to_keypoint(keypoints)


            # Convert keypoints to PyTorch tensor
            keypoints = torch.tensor(keypoints).float()

            # keep only the keypoints of the person with the highest confidence
            if keypoints.shape[0] > 1:
                keypoints = keypoints[torch.argmax(keypoints[:, 6]), :]
            # if only one set of keypoints is detected, keep them
            elif keypoints.shape[0] == 1:
                keypoints = keypoints[0, :]
            # if no keypoints are detected, skip frame
            else:
                continue

            # normalize keypoints to the center of the image
            xchange = 640/2 - keypoints[2]
            ychange = 384/2 - keypoints[3]

            # get rid of the first 7 elements of the keypoints tensor
            keypoints = keypoints[7:]
            # normalize keypoints to the center of the image
            for idx, point in enumerate(keypoints):
                if idx % 3 == 0:
                    keypoints[idx] = point + xchange
                elif idx % 3 == 1:
                    keypoints[idx] = point + ychange

            ## Uncomment for keypoints visualization
            # blank = np.zeros((384, 640, 3), np.uint8)
            # for i in range(0, len(keypoints), 3):
            #     cv2.circle(blank, (int(keypoints[i]), int(keypoints[i+1])), 3, (255, 0, 0), -1)

            # cv2.imshow("frame", blank)
            # cv2.waitKey(1)


            # Save keypoints
            keypoints_list.append(keypoints)

        # Release video capture
        cap.release()

        # Convert list to PyTorch tensor
        keypoints_tensor = torch.stack(keypoints_list)


        # Save keypoints to file
        # torch.save(keypoints_tensor, os.path.join("../keypoints_test_norm", video_filename + ".pt"))

        # torch.save(keypoints_tensor, os.path.join("../keypoints_norm", video_filename + ".pt"))

        # torch.save(keypoints_tensor, os.path.join("../keypoints_test", video_filename + ".pt"))

        # torch.save(keypoints_tensor, os.path.join("../keypoints2.0", video_filename + ".pt"))
