# Purpose: extract normalised frames from dataset
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
    # print(image.shape)
    image = letterbox(image, new_shape = (640), stride=64, auto=True)[0] # shape: (567, 960, 3)
    # print(image.shape) # 384x640
    # Apply transforms
    image = transforms.ToTensor()(image) 
    if torch.cuda.is_available():
        image = image.half().to(device)
    # Turn image into batch
    image = image.unsqueeze(0)
    with torch.no_grad():
        output, _ = model(image)
    return output

folder = [
        #   '../dataset/train/tr_underwater/tr_u_drown', 
        #   '../dataset/train/tr_underwater/tr_u_swim', 
        #   '../dataset/train/tr_underwater/tr_u_misc', 
        #   '../dataset/train/tr_underwater/tr_u_idle',
        #   '../dataset/train/tr_overhead/tr_o_drown',
        #   '../dataset/train/tr_overhead/tr_o_swim',
        #   '../dataset/train/tr_overhead/tr_o_misc',
        #   '../dataset/train/tr_overhead/tr_o_idle'
          '../dataset/test/te_underwater',
          '../dataset/test/te_overhead'
        ]

# Loop over videos in the folder
# for directory in os.listdir(folder):
for directory in folder:
    for video_filename in os.listdir(os.path.join(directory)):
        # num = video_filename.split("_")[3]
        # nums = ["1.mp4","3.mp4","4.mp4","5.mp4","7.mp4","9.mp4","13.mp4","16.mp4","18.mp4"]
        if not video_filename.endswith(".mp4"):
            continue

        # Load video
        cap = cv2.VideoCapture(os.path.join(directory, video_filename))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # out = cv2.VideoWriter(os.path.join("../train_frames", video_filename), fourcc, 30.0, (120,120))
        out = cv2.VideoWriter(os.path.join("../test_frames", video_filename), fourcc, 30.0, (120,120))


        # Loop over video frames
        while True:        
            os.system('cls' if os.name == 'nt' else 'clear')
            print("Processing video:", video_filename)
            print("Frame:", cap.get(cv2.CAP_PROP_POS_FRAMES), "/", cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # print video num out of total num
            print("Video:", os.listdir(directory).index(video_filename) + 1, "/", len(os.listdir(directory)))

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
            # print(keypoints.shape)
            # print(keypoints.type())

            if keypoints.shape[0] > 1:
                # # keep only the keypoints of the person with the highest confidence
                # keypoints = keypoints[torch.argmax(keypoints[:, 6]), :]
                max_conf = 0
                for person in keypoints:
                    conf_count = 0
                    for i in range(9, len(person), 3):
                        if person[i] > 0.5:
                            conf_count += 1
                    if conf_count > max_conf:
                        max_conf = conf_count
                        keypoints = person

                if max_conf == 0:
                    keypoints = keypoints[torch.argmax(keypoints[:, 6]), :]


            elif keypoints.shape[0] == 1:
                keypoints = keypoints[0, :]
            else:
                continue

            # # normalize keypoints to the center of the image
            # xchange = 640/2 - keypoints[2]
            # ychange = 384/2 - keypoints[3]

            bbwidth = keypoints[4] if keypoints[4] > keypoints[5] else keypoints[5]

            xchange = keypoints[2] - bbwidth/2
            ychange = keypoints[3] - bbwidth/2

            keypoints = keypoints[7:]
            for idx, point in enumerate(keypoints):
                if idx % 3 == 0:
                    # keypoints[idx] = (point + xchange)/6.4
                    keypoints[idx] = (point - xchange)/(bbwidth/100)+10

                elif idx % 3 == 1:
                    # keypoints[idx] = (point + ychange)/3.84
                    keypoints[idx] = (point - ychange)/(bbwidth/100)+10

            # blank = np.zeros((384, 640, 3), np.uint8)
            blank = np.zeros((120, 120, 3), np.uint8)
            # cv2.rectangle(blank, (10, 10), (110, 110), (255, 0, 0), -1)


            for i in range(0, len(keypoints), 3):
                # cv2.circle(blank, (int(keypoints[i]), int(keypoints[i+1])), 1, (255, 0, 0), -1)
                plot_skeleton_kpts(blank, keypoints, 3)

            out.write(blank)

            # cv2.imshow("frame", blank)
            # cv2.waitKey(1)


        # Release video capture
        cap.release()
        out.release()


        # Save keypoints to file
        # torch.save(keypoints_tensor, os.path.join("../keypoints_test_norm", video_filename + ".pt"))

        # torch.save(keypoints_tensor, os.path.join("../keypoints_norm", video_filename + ".pt"))

        # torch.save(keypoints_tensor, os.path.join("../keypoints_test", video_filename + ".pt"))

        # torch.save(keypoints_tensor, os.path.join("../keypoints2.0", video_filename + ".pt"))
