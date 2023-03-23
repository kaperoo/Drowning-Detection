import torch
from torchvision import transforms
from os.path import dirname, join

from cnnmodel import CNNModel

import sys
sys.path.append(join(dirname(__file__), "../yolov7"))

from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt, xywh2xyxy
from utils.plots import output_to_keypoint, plot_skeleton_kpts, plot_one_box

import matplotlib.pyplot as plt
import cv2
import numpy as np
import os


# Load YOLOv7 model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

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
    return output, image

def draw_box_and_pred(image, pred, file):

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

    if pred == 0:
        inf_text = 'drowning'
    elif pred == 1:
        inf_text = 'swimming'
    elif pred == 2:
        inf_text = 'miscellaneous'
    elif pred == 3:
        inf_text = 'idle'

    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    for idx in range(kpdisplay.shape[0]):
        plot_skeleton_kpts(nimg, kpdisplay[idx, 7:].T, 3)
    
    boxcoords = xywh2xyxy(kpdisplay[:, 2:6])
    colour = (0, 100, 0) if pred == labelid else (0, 0, 100)
    plot_one_box(boxcoords[0], nimg, label=inf_text, color=colour, line_thickness=2)
    cv2.imshow('frame', nimg)
    cv2.waitKey(1)


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

cnnmodel = CNNModel().to('cuda:0')
with open('model_state.pt', 'rb') as f:
    cnnmodel.load_state_dict(torch.load(f))

# Loop over videos in the folder
# for directory in os.listdir(folder):
for directory in folder:
    for video_filename in os.listdir(os.path.join(directory)):
        if not video_filename.endswith(".mp4"):
            continue

        # Load video
        cap = cv2.VideoCapture(os.path.join(directory, video_filename))
    
        # Loop over video frames
        while True:        
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLOv7 on the frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output, image = run_inference(frame)

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

            kpdisplay = keypoints.clone()

            if keypoints.shape[0] > 1:
                # keep only the keypoints of the person with the highest confidence
                keypoints = keypoints[torch.argmax(keypoints[:, 6]), :]
            elif keypoints.shape[0] == 1:
                keypoints = keypoints[0, :]
            else:
                continue

            # normalize keypoints to the center of the image
            xchange = 640/2 - keypoints[2]
            ychange = 384/2 - keypoints[3]

            keypoints = keypoints[7:]
            for idx, point in enumerate(keypoints):
                if idx % 3 == 0:
                    keypoints[idx] = point + xchange
                elif idx % 3 == 1:
                    keypoints[idx] = point + ychange

            inf_output = torch.argmax(cnnmodel(keypoints.unsqueeze(0).to('cuda:0')))
            if inf_output == 0:
                inf_text = 'drowning'
            elif inf_output == 1:
                inf_text = 'swimming'
            elif inf_output == 2:
                inf_text = 'miscellaneous'
            elif inf_output == 3:
                inf_text = 'idle'

            draw_box_and_pred(image, inf_output, video_filename)
        # Release video capture
        cv2.destroyAllWindows()
        cap.release()