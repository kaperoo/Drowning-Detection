# #read videos from C:\Users\User\Desktop\Code\FYP\train_frames_no_misc and divide them into 5 frames each
# #save them to C:\Users\User\Desktop\Code\FYP\train_frames_no_misc_5

import cv2
import os

# Set the directory path where your videos are stored
video_dir = "C:\\Users\\User\\Desktop\\Code\\FYP\\test_frames"

# Set the directory path where you want to save the 5-frame video segments
output_dir = "C:\\Users\\User\\Desktop\\Code\\FYP\\test_frames_10"

# Loop over each video file in the directory
for filename in os.listdir(video_dir):

    # Load the video file
    video_path = os.path.join(video_dir, filename)
    video = cv2.VideoCapture(video_path)

    # Get the video length and calculate the number of 5-frame segments
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    num_segments = num_frames // 10

    # Loop over each 5-frame segment and save it as a new video file
    for i in range(num_segments):

        # Set the start and end frames of the segment
        start_frame = i * 10
        end_frame = start_frame + 9

        # Set the output file name
        output_name = f"{filename}_{i}.mp4"

        # Create a VideoWriter object to save the segment
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = os.path.join(output_dir, output_name)
        writer = cv2.VideoWriter(output_path, fourcc, 30.0, (120, 120))

        # Loop over each frame in the segment and write it to the output file
        for j in range(start_frame, end_frame+1):
            video.set(cv2.CAP_PROP_POS_FRAMES, j)
            ret, frame = video.read()
            if ret:
                writer.write(frame)

        # Release the VideoWriter object
        writer.release()

    # Release the VideoCapture object
    video.release()

# import cv2
# import os

# # Set the directory path where your videos are stored
# video_dir = "C:\\Users\\User\\Desktop\\Code\\FYP\\train_frames"

# # Set the directory path where you want to save the frames
# output_dir = "C:\\Users\\User\\Desktop\\Code\FYP\\train_frames_1"

# # Loop over each video file in the directory
# for filename in os.listdir(video_dir):

#     # Load the video file
#     video_path = os.path.join(video_dir, filename)
#     video = cv2.VideoCapture(video_path)

#     # Loop over each frame in the video and save it as a separate image file
#     frame_num = 0
#     while True:
#         ret, frame = video.read()
#         if not ret:
#             break

#         # Set the output file name
#         output_name = f"{filename}_frame{frame_num}.jpg"

#         # Write the frame to the output file
#         output_path = os.path.join(output_dir, output_name)
#         cv2.imwrite(output_path, frame)

#         frame_num += 1

#     # Release the VideoCapture object
#     video.release()