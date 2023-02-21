import torch
from torchvision import transforms
from os.path import dirname, join

import sys
sys.path.append(join(dirname(__file__), "../yolov7"))

from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts, plot_one_box

import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def load_model():
    model = torch.load(join(dirname(__file__), "../yolov7/yolov7-w6-pose.pt"), map_location=device)['model']
    # Put in inference mode
    model.float().eval()

    if torch.cuda.is_available():
        # half() turns predictions into float16 tensors
        # which significantly lowers inference time
        model.half().to(device)
    return model

model = load_model()

def run_inference(image):
    # Resize image to 640x368 and pad
    image = letterbox(image, 640, stride=64, auto=True)[0] # shape: (368, 640, 3)
    # image = letterbox(image, 960, stride=64, auto=True)[0] # shape: (567, 960, 3)
    
    # Apply transforms
    image = transforms.ToTensor()(image) # torch.Size([3, 567, 960])
    # image = torch.tensor(np.array([image.numpy()]))
    if torch.cuda.is_available():
        image = image.half().to(device)
    # output, _ = model(image)
    # Turn image into batch
    image = image.unsqueeze(0) # torch.Size([1, 3, 567, 960])
    with torch.no_grad():
        output, _ = model(image)
    return output, image

def draw_keypoints(output, image):
    output = non_max_suppression_kpt(output, 
                                     0.25, # Confidence Threshold
                                     0.65, # IoU Threshold
                                     nc=model.yaml['nc'], # Number of Classes
                                     nkpt=model.yaml['nkpt'], # Number of Keypoints
                                     kpt_label=True)
    with torch.no_grad():
        output = output_to_keypoint(output)

    print(output[0, 0:7])

    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    for idx in range(output.shape[0]):
        plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)
        # plot_one_box(output[idx, 0:4], nimg, label='Drowning', color=(255, 0, 0), line_thickness=3)

    return nimg

# read videos in a folder and save them in one video
def pose_estimation_video(folder):
    # VideoWriter for saving the video
    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # out = cv2.VideoWriter('test7.mp4', fourcc, 30.0, (640, 368), True)
    for filename in os.listdir(folder):
        if filename.endswith(".mp4"):
            print(filename)
            cap = cv2.VideoCapture(os.path.join(folder, filename))
            while cap.isOpened():
                (ret, frame) = cap.read()
                if ret == True:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    output, image = run_inference(frame)
                    image = draw_keypoints(output, image)
                    frame = cv2.resize(image, (960, 567))
                    # out.write(frame)
                    cv2.imshow('Pose estimation', frame)
                else:
                    break

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            cap.release()
        else:
            continue

    # out.release()
    cv2.destroyAllWindows()

# # read frames from folder and save them in a video
# def pose_estimation_video(folder):
#     # VideoWriter for saving the video
#     fourcc = cv2.VideoWriter_fourcc(*'MP4V')
#     out = cv2.VideoWriter('drowning.mp4', fourcc, 30.0, (960, 567))
#     for filename in os.listdir(folder):
#         if filename.endswith(".jpg"):
#             image = cv2.imread(os.path.join(folder, filename))
#             output, image = run_inference(image)
#             image = draw_keypoints(output, image)
#             out.write(image)
#             # cv2.imshow('Pose estimation', image)
#         else:
#             continue

#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break

#     out.release()
#     cv2.destroyAllWindows()
folder = '../dataset/train/tr_underwater/tr_u_drown'
pose_estimation_video(folder)