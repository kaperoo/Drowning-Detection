# PURPOSE: Integrated System for Drowning Detection with CNNSEQ model
import torch
from torchvision import transforms
from os.path import dirname, join


import sys
sys.path.append(join(dirname(__file__), "..\\yolov7"))
sys.path.append(join(dirname(__file__), ".\\models"))
from cnnseq import CNNSEQ

from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt, xywh2xyxy
from utils.plots import output_to_keypoint, plot_skeleton_kpts, plot_one_box

import cv2
import numpy as np
import os

# Load YOLOv7 model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

# function for drawing bounding box and prediction
def draw_box_and_pred(image, pred, conf):

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
    else:
        inf_text = ''

    # draw the bounding box on the image
    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    boxcoords = xywh2xyxy(kpdisplay[:, 2:6])
    plot_one_box(boxcoords[0], nimg, label=str(inf_text + " " + str(conf)), color=colour, line_thickness=2)

    # display the image
    cv2.imshow('frame', nimg)
    cv2.waitKey(1)

# path to video footage
footage = sys.argv[1]

# set max sequence length
seq_length = 30

# load cnnseq model
cnnmodel = CNNSEQ().to('cuda:0')
with open('models\\model_cnnseq.pt', 'rb') as f:
    cnnmodel.load_state_dict(torch.load(f))

if not footage.endswith(".mp4"):
    print("Not a video file")
    exit()
# Load video
cap = cv2.VideoCapture(footage)
# Create a buffer for the frames
frame_buffer = []
inf_output = None
# Loop over frames
while True:  
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv7 on the frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output, image = run_inference(frame)

    # Extract keypoints from the output and perform non-max suppression
    keypoints = []
    keypoints = non_max_suppression_kpt(output, 
                                0.25, # Confidence Threshold
                                0.65, # IoU Threshold
                                nc=model.yaml['nc'], # Number of Classes
                                nkpt=model.yaml['nkpt'], # Number of Keypoints
                                kpt_label=True)

    # get keypoints from filtered output
    with torch.no_grad():
        keypoints = output_to_keypoint(keypoints)

    # Convert keypoints to PyTorch tensor
    keypoints = torch.tensor(keypoints).float()

    # clone keypoints to keep the first 7 columns
    kpdisplay = keypoints.clone()

    # if there are more than one person detected, keep only the person with the highest confidence
    if keypoints.shape[0] > 1:
        keypoints = keypoints[torch.argmax(keypoints[:, 6]), :]
    # if there is only one person detected, keep the person
    elif keypoints.shape[0] == 1:
        keypoints = keypoints[0, :]
    # if there are no person detected, skip the frame
    else:
        continue

    # normalize keypoints to the center of the image
    xchange = 640/2 - keypoints[2]
    ychange = 384/2 - keypoints[3]

    # get rid of the first 7 columns
    keypoints = keypoints[7:]
    # normalize keypoints to the center of the image
    for idx, point in enumerate(keypoints):
        if idx % 3 == 0:
            keypoints[idx] = point + xchange
        elif idx % 3 == 1:
            keypoints[idx] = point + ychange

    # prepare keypoints for cnnseq model
    keypoints = keypoints.reshape(17, 3)
    frame_buffer.append(keypoints)

    # keep only the last 30 frames
    if len(frame_buffer) > seq_length:
        frame_buffer.pop(0)

    # run cnnseq model if there are more than 2 frames in the buffer
    if len(frame_buffer) > 2:
        # convert frame buffer to tensor
        keypoints_tensor = torch.stack(frame_buffer, dim=0).unsqueeze(1).to(device)
        keypoints_tensor = keypoints_tensor.reshape(1, len(frame_buffer), 17, 3, 1)
        # run cnnseq model
        with torch.no_grad():
            output = cnnmodel(keypoints_tensor)

    # Get the predicted class
    _, predicted_class = torch.max(output, 2)
    predicted_class = predicted_class.squeeze().cpu().numpy()

    # get the confidence of the prediction
    softmax = torch.nn.functional.softmax(output, dim=2)
    conf = torch.max(softmax, dim=2).values

    # average confidence over the sequence
    conf = torch.mean(conf).item()
    # round confidence to 2 decimal places
    conf = round(conf, 2)

    # Get the label of the most frequent class in the sequence
    inf_output = np.argmax(np.bincount(predicted_class))

    # draw the bounding box and the prediction on the image            
    draw_box_and_pred(image, inf_output, conf)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    
# Release video capture
cv2.destroyAllWindows()
cap.release()