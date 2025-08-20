from datasets import load_dataset, Value
import cv2
import numpy as np
import os
import imageio.v3 as iio
from tqdm import tqdm
import json
# import torch
# from model.qwen_model import QwenModel
# from dataloader_utils.dataloader_waymo_train import WaymoE2EDatasetTraining, train_collate_fn
# from tqdm import tqdm
# from torch.utils.data import DataLoader
# import os
# import numpy as np

#model = QwenModel(model_to_use=72, device='auto')
# training_dataset = WaymoE2EDatasetTraining('/cluster/scratch/arsood/data_clean', 4)
# training_loader = DataLoader(training_dataset, batch_size=1, shuffle=False, collate_fn=train_collate_fn)
#dir_to_save = '/cluster/scratch/arsood/data_strings_covla'



import matplotlib.pyplot as plt
import numpy as np

def device_to_camera(P_device, extrinsic_matrix):
    """Convert device coordinates to camera coordinates."""
    P_device_hom = np.append(P_device, 1)
    P_camera_hom = np.dot(extrinsic_matrix, P_device_hom)
    return P_camera_hom[:3]

def camera_to_image(P_camera, intrinsic_matrix):
    """Convert camera coordinates to image coordinates."""
    P_image_homogeneous = np.dot(intrinsic_matrix, P_camera)
    P_image = P_image_homogeneous[:2] / P_image_homogeneous[2]
    return P_image

def plot_trajectory_on_image(
    frame, trajectory, extrinsic_matrix, intrinsic_matrix, marker="o", color="red"
):
    # Convert device coordinates to camera coordinates
    future_pos_camera = np.array(
        [device_to_camera(p, extrinsic_matrix) for p in trajectory]
    )
    # Keep only points in front of the camera (z > 0)
    future_pos_camera = future_pos_camera[future_pos_camera[:, 2] > 0]

    # Convert camera coordinates to image coordinates
    future_pos_image = np.array(
        [camera_to_image(p, intrinsic_matrix) for p in future_pos_camera]
    )

    # Filter out points that are outside the image bounds
    image_height, image_width = frame.shape[:2]
    future_pos_image = future_pos_image[
        (future_pos_image[:, 0] >= 0) & (future_pos_image[:, 0] < image_width) &
        (future_pos_image[:, 1] >= 0) & (future_pos_image[:, 1] < image_height)
    ]

    plt.imshow(frame)
    plt.axis("off")
    if len(future_pos_image) > 0:
        plt.plot(
            future_pos_image[:, 0], future_pos_image[:, 1],
            marker=marker, color=color, linestyle="solid", linewidth=1, markersize=3,
        )
    plt.savefig('/cluster/home/arsood/Semester_Project_Official/fig.png')


dir_path_videos = '/cluster/scratch/arsood/covla/videos'
dir_path_states  = '/cluster/scratch/arsood/covla/states'
filenames_states = os.listdir(dir_path_states)
filename_videos = os.listdir(dir_path_videos)
filenames_states = sorted(filenames_states)
filename_videos = sorted(filename_videos)


data = []
with open(os.path.join(dir_path_states,filenames_states[0]), 'r') as f:
    for line in f:
        if line.strip():  # skip empty lines
            data.append(json.loads(line))
# for i in tqdm(range(len(filenames))):
#     video_array = iio.imread(os.path.join(dir_path, filenames[i]), index=None)
#     if(video_array.shape[0] != 600):
#         print("No")
#print(len(data))      # number of JSON objects
#print(data[1]['1']['trajectory']) 
video_array = iio.imread(os.path.join(dir_path_videos,filename_videos[0]), index=None)[0]

plot_trajectory_on_image(
    frame=video_array,
    trajectory=data[0]['0']['trajectory'],
    extrinsic_matrix=np.array(data[0]['0']["extrinsic_matrix"]),
    intrinsic_matrix=np.array(data[0]['0']["intrinsic_matrix"]),
    marker="o",
    color="red"
)
plt.figure()
plt.plot(np.array(data[0]['0']['trajectory'])[...,1], np.array(data[0]['0']['trajectory'])[...,0], color='blue', marker='o')
plt.savefig('/cluster/home/arsood/Semester_Project_Official/fig2.png')
