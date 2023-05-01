# PURPOSE: Display inference visually
import torch
from torchvision import transforms
from os.path import dirname, join

from cnnrnn import CNNModel

import sys
sys.path.append(join(dirname(__file__), "..\\yolov7"))

from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt, xywh2xyxy
from utils.plots import output_to_keypoint, plot_skeleton_kpts, plot_one_box

import cv2
import numpy as np
import os

import time

# Load YOLOv7 model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

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

def draw_box_and_pred(image, pred, file, conf):

    label = file.split("_")[2]
    labelid = 4
    colour = (100, 100, 100)
    if label == "drown":
        labelid = 0
    elif label == "swim":
        labelid = 1
    elif label == "idle":
        labelid = 2

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

    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    # for idx in range(kpdisplay.shape[0]):
    #     plot_skeleton_kpts(nimg, kpdisplay[idx, 7:].T, 3)
    
    boxcoords = xywh2xyxy(kpdisplay[:, 2:6])
    # colour = (0, 100, 0) if pred == labelid else (0, 0, 100)
    plot_one_box(boxcoords[0], nimg, label=str(inf_text + " " + str(conf)), color=colour, line_thickness=2)
    
    return nimg


folder = [
        #   '../dataset/train/tr_underwater/tr_u_drown', 
        #   '../dataset/train/tr_underwater/tr_u_swim', 
        #   '../dataset/train/tr_underwater/tr_u_misc', 
        #   '../dataset/train/tr_underwater/tr_u_idle',
        #   '../dataset/train/tr_overhead/tr_o_drown',
        #   '../dataset/train/tr_overhead/tr_o_swim',
        #   '../dataset/train/tr_overhead/tr_o_misc',
        #   '../dataset/train/tr_overhead/tr_o_idle'
          '..\\dataset\\test\\te_underwater',
          '..\\dataset\\test\\te_overhead'
        ]

seq_length = 30

cnnmodel = CNNModel().to('cuda:0')
with open('model_state_rnn.pt', 'rb') as f:
    cnnmodel.load_state_dict(torch.load(f))

# Loop over videos in the folder
# for directory in os.listdir(folder):
total_change = 0
total_time = 0
for directory in folder:
    for video_filename in os.listdir(os.path.join(directory)):
        if not video_filename.endswith(".mp4"):
            continue

        

        # Load video
        cap = cv2.VideoCapture(os.path.join(directory, video_filename))

        #get size of the frames
        if cap.get(cv2.CAP_PROP_FRAME_HEIGHT) == 1440:
            hsize = 512
        else:
            hsize = 384
                   

        # save video
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # out = cv2.VideoWriter(video_filename.split(".")[0]+'.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (640, hsize))
        frame_buffer = []
        inf_output = None
        # Loop over video frames
        fr = 0
        # times = [0,0,0,0,0]
        prev_class = -1


        while True:  
            fr += 1      
            ret, frame = cap.read()
            if not ret:
                break

            # # start measuring time for yolo inference
            # yolo_start = time.time()

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

            # # end measuring time for yolo inference
            # yolo_end = time.time()

            # # start measuring time for data preprocessing
            # data_start = time.time()

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

            # # end measuring time for data preprocessing
            # data_end = time.time()

            # # start measuring time for cnn inference
            # cnn_start = time.time()

            keypoints = keypoints.reshape(17, 3)
            frame_buffer.append(keypoints)

            if len(frame_buffer) > seq_length:
                frame_buffer.pop(0)

            # if len(frame_buffer) == seq_length:
            if len(frame_buffer) > 2:
            # Convert keypoints to tensor and add batch dimension
            # for i in frame_buffer:
                # print(i.shape)
                keypoints_tensor = torch.stack(frame_buffer, dim=0).unsqueeze(1).to(device)
                keypoints_tensor = keypoints_tensor.reshape(1, len(frame_buffer), 17, 3, 1)
                # print(keypoints_tensor.shape)
                # Forward pass through the model
                with torch.no_grad():
                    output = cnnmodel(keypoints_tensor)

            # Get the predicted class
            inf_output = np.argmax(output.cpu().numpy())

            # get the confidence of the prediction
            softmax = torch.nn.functional.softmax(output, dim=2)
            conf = torch.max(softmax, dim=2).values

            # average confidence over the sequence
            conf = torch.mean(conf).item()
            # round confidence to 2 decimal places
            conf = round(conf, 2)

            if fr == 1:
                prev_class = inf_output
            elif inf_output != prev_class:
                prev_class = inf_output
                total_change += 1

            # # end measuring time for cnn inference
            # cnn_end = time.time()

            # # start measuring time for drawing
            # draw_start = time.time()
            nimg = draw_box_and_pred(image, inf_output, video_filename, conf)
            # cv2.imshow('frame', nimg)
            print(nimg.shape)
            cv2.waitKey(1)
            # out.write(nimg)
            # # end measuring time for drawing
            # draw_end = time.time()
            
            # times[0] += yolo_end - yolo_start
            # times[1] += data_end - data_start
            # times[2] += cnn_end - cnn_start
            # times[3] += draw_end - draw_start
            # times[4] += draw_end - yolo_start

            # if fr % 30 == 0:
            #     print('\nYolo inference time: ', times[0])
            #     print('Data preprocessing time: ', times[1])
            #     print('CNN inference time: ', times[2])
            #     print('Drawing time: ', times[3])
            #     print('Total time: ', times[4])
            #     print('FPS: ', 1/(times[4]/30))
            #     times = [0,0,0,0,0]

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            
        # Release video capture
        cv2.destroyAllWindows()
        cap.release()
        # out.release()

        total_time = total_time + fr
        print('Total changes: ', (total_change/total_time)*30)
        print('Avg time of class', total_time/total_change)