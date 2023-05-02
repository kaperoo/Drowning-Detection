# PURPOSE: Final Integrated System for Drowning Detection with Baseline model
import torch
from torchvision import transforms
from os.path import dirname, join


import sys
sys.path.append(join(dirname(__file__), "..\\..\\yolov7"))
sys.path.append(join(dirname(__file__), "..\\models"))

from baseline import Baseline

from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt, xywh2xyxy
from utils.plots import output_to_keypoint, plot_skeleton_kpts, plot_one_box

import cv2
import numpy as np
import os


# Load YOLOv7 model on GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the yolo model
def load_model():
    model = torch.load(join(dirname(__file__), "..\\yolov7\\yolov7-w6-pose.pt"), map_location=device)['model']
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
    return output, image

def draw_box_and_pred(image, pred, file, conf):

    # set the colour of the bounding box
    # also set the label text
    colour = (100, 100, 100)
    if pred == 0:
        inf_text = 'drowning'
        colour = (0, 0, 100)
    elif pred == 1:
        inf_text = 'swimming'
        colour = (0, 100, 0)
    elif pred == 2:
        inf_text = 'idle'
        colour = (0, 100, 100)

    # prepare the image for display
    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)

    # get the bounding box coordinates    
    boxcoords = xywh2xyxy(kpdisplay[:, 2:6])
    # draw the bounding box and label
    plot_one_box(boxcoords[0], nimg, label=str(inf_text + " " + str(conf)), color=colour, line_thickness=2)
    
    # display the image
    cv2.imshow('frame', nimg)
    cv2.waitKey(1)


# define the folder with input footage
folder = [
          '..\\dataset\\test\\te_underwater',
          '..\\dataset\\test\\te_overhead'
        ]

# load the baseline model
cnnmodel = Baseline().to('cuda:0')
with open('model_baseline.pt', 'rb') as f:
    cnnmodel.load_state_dict(torch.load(f))

# Loop over videos in the folder
for directory in folder:
    for video_filename in os.listdir(os.path.join(directory)):
        if not video_filename.endswith(".mp4"):
            continue

        # Load video
        cap = cv2.VideoCapture(os.path.join(directory, video_filename))
    
        # Loop over video frames
        fr = 0
        while True:      
            fr += 1  
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLOv7 on the frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output, image = run_inference(frame)

            # Extract keypoints from the output
            # and perform non-max suppression
            keypoints = []
            keypoints = non_max_suppression_kpt(output, 
                                        0.25, # Confidence Threshold
                                        0.65, # IoU Threshold
                                        nc=model.yaml['nc'], # Number of Classes
                                        nkpt=model.yaml['nkpt'], # Number of Keypoints
                                        kpt_label=True)

            # get the keypoints for display
            with torch.no_grad():
                keypoints = output_to_keypoint(keypoints)

            # Convert keypoints to PyTorch tensor
            keypoints = torch.tensor(keypoints).float()

            # clone the keypoints to keep the first 7 values
            kpdisplay = keypoints.clone()

            # keep only the keypoints with the highest confidence
            if keypoints.shape[0] > 1:
                keypoints = keypoints[torch.argmax(keypoints[:, 6]), :]
            # if only one set of keypoints, keep them
            elif keypoints.shape[0] == 1:
                keypoints = keypoints[0, :]
            # if no keypoints, continue to next frame
            else:
                continue

            # normalize keypoints to the center of the image
            xchange = 640/2 - keypoints[2]
            ychange = 384/2 - keypoints[3]

            # get rid of the first 7 values
            keypoints = keypoints[7:]
            # normalize the keypoints
            for idx, point in enumerate(keypoints):
                if idx % 3 == 0:
                    keypoints[idx] = point + xchange
                elif idx % 3 == 1:
                    keypoints[idx] = point + ychange

            # predict the class of the keypoints
            pred = cnnmodel(keypoints.unsqueeze(0).to('cuda:0'))

            # get the class of the prediction
            inf_output = torch.argmax(pred)
            # get the confidence of the prediction
            softmax = torch.nn.functional.softmax(pred, dim=1)
            conf = torch.max(softmax, dim=1).values.item()
            # round confidence to 2 decimal places
            conf = round(conf, 2)

            # draw the bounding box and label
            draw_box_and_pred(image, inf_output, video_filename, conf)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        # Release video capture
        cv2.destroyAllWindows()
        cap.release()