import torch
from torch.utils.data import Dataset
import os
import random
import numpy as np
import cv2
import imageio.v3 as iio
import json
from .strings import training_prompt_covla
import random
from scipy.interpolate import CubicSpline
import math



import matplotlib.pyplot as plt
import numpy as np



class CovlaDatasetTraining(Dataset):
    def __init__(self, data_path_videos, data_path_states, indices, seq_len = 4, self_cur_idx = 7):
        super().__init__()
        self.data_path_videos = data_path_videos
        self.list_videos = os.listdir(self.data_path_videos)
        self.list_videos = sorted(self.list_videos)

        self.data_path_states = data_path_states
        self.list_states = os.listdir(self.data_path_states)
        self.list_states = sorted(self.list_states)

        self.list_videos = [self.list_videos[idx] for idx in indices]
        self.list_states = [self.list_states[idx] for idx in indices]

        self.window = 40
        self.seq_len = seq_len
        self.self_cur_idx = self_cur_idx

    def __len__(self):
        return len(self.list_states)
    
    def __getitem__(self, index):
        file_path_video = os.path.join(self.data_path_videos, self.list_videos[index])
        file_path_states = os.path.join(self.data_path_states, self.list_states[index])
        resize_factor = 4

        path_name = os.path.splitext(self.list_videos[index])[0]

        interval = 80+self.window*self.self_cur_idx
        candidate_interval = list(range(interval, interval+40, 2))
        start_point = random.choice(candidate_interval)

        video_array = iio.imread(file_path_video, index=None)
        states_array = []
        intent_traj = "GO_STRAIGHT"
        with open(file_path_states, 'r') as f:
            for line in f:
                if line.strip():
                    states_array.append(json.loads(line))

        trajectory = np.array(states_array[start_point][str(start_point)]['trajectory'])

        

        traj_future = np.array(states_array[start_point+60][str(start_point+60)]['trajectory'])[0:40,0:2]
        traj_past = np.array(states_array[start_point-60][str(start_point-60)]['trajectory'])[...,0:2]
        traj_past_first = np.array(states_array[start_point-80][str(start_point-80)]['trajectory'])[0:20,0:2]


        l2_past_first = np.array(states_array[start_point-80][str(start_point-80)]['orientations_ned'])
        l1_past_first = np.array(states_array[start_point-60][str(start_point-60)]['orientations_ned'])
        angle_past_rad_first = l2_past_first[-1]-l1_past_first[-1]
        #angle_past_rad = math.atan((traj_past[-1][1]-traj_past[-2][1])/(traj_past[-1][0]-traj_past[-2][0]))
        rot_mat_past_first = np.array([
            [math.cos(angle_past_rad_first),-math.sin(angle_past_rad_first)],
            [math.sin(angle_past_rad_first),math.cos(angle_past_rad_first)]
        ])
        traj_past_first = traj_past_first-traj_past_first[-1]
        transform_points_past_first = np.matmul(traj_past_first, rot_mat_past_first)

        traj_past = np.vstack([transform_points_past_first,traj_past])



        l2_past = np.array(states_array[start_point-60][str(start_point-60)]['orientations_ned'])
        l1_past = np.array(states_array[start_point][str(start_point)]['orientations_ned'])
        angle_past_rad = l2_past[-1]-l1_past[-1]
        #angle_past_rad = math.atan((traj_past[-1][1]-traj_past[-2][1])/(traj_past[-1][0]-traj_past[-2][0]))
        rot_mat_past = np.array([
            [math.cos(angle_past_rad),-math.sin(angle_past_rad)],
            [math.sin(angle_past_rad),math.cos(angle_past_rad)]
        ])
        traj_past = traj_past-traj_past[-1]
        transform_points_past = (np.matmul(traj_past, rot_mat_past)[::-5])[::-1]


        l2 = np.array(states_array[start_point+60][str(start_point+60)]['orientations_ned'])
        l1 = np.array(states_array[start_point][str(start_point)]['orientations_ned'])
        angle_rad = l2[-1]-l1[-1]
        #angle_rad = math.atan((traj_present[1][1]-traj_present[0][1])/(traj_present[1][0]-traj_present[0][0]))
        rot_mat = np.array([
            [math.cos(angle_rad),-math.sin(angle_rad)],
            [math.sin(angle_rad),math.cos(angle_rad)]
        ])
        transform_points = np.matmul(traj_future, rot_mat)
        transform_points = np.array(states_array[start_point][str(start_point)]['trajectory'])[-1, 0:2]+transform_points
        transform_points = np.vstack([trajectory[...,0:2],transform_points])
        transform_points = np.vstack([transform_points[5::5], transform_points[-1]])

        video_array_1 = video_array[(start_point-7):start_point+1]
        video_array_1 = (video_array_1[::-2])[::-1]

        video_array_resize = [cv2.resize(video_array_1[j], (int(video_array_1[j].shape[1]/resize_factor), int(video_array_1[j].shape[0]/resize_factor)), interpolation=cv2.INTER_AREA) for j in range(0,video_array_1.shape[0])]
        video_array_resize = np.array(video_array_resize)

        # plt.figure()
        # plt.imshow(video_array_resize[-1])
        # plt.savefig('/cluster/home/arsood/Semester_Project_Official/fig_2.png')
        
        

        if(states_array[start_point][str(start_point)]['rightBlinker'] == True):
            intent_traj = 'GO_RIGHT'
        elif (states_array[start_point][str(start_point)]['leftBlinker'] == True):
            intent_traj = 'GO_LEFT'

        # plt.figure()
        # plt.xlim([-20,20])
        # #plt.plot(trajectory[...,1], trajectory[...,0], color='red')
        # plt.plot(transform_points[...,1], transform_points[...,0], color='blue')
        # plt.plot(transform_points_past[..., 1], transform_points_past[..., 0], color='green')
        # plt.savefig('/cluster/home/arsood/Semester_Project_Official/fig_2.png')
        
        path_file_name = f"{path_name}-{start_point}"

        np.set_printoptions(precision=4, suppress=True)
        prompt_to_use = training_prompt_covla(np.array_str(transform_points_past), intent_traj, np.array_str(transform_points))
        message_to_pass = [{
        "role": "user",
        "content": [
            {"type": "video", "video": ""},
            {"type": "text", "text": prompt_to_use },
        ],
        }
        # ,{
        #     "role": "assistant",
        #     "content": [{"type": "text", "text": "{"+np.array_str(next_state_traj_5[...,:2])+"}"}],
        # }
        ]

        return video_array_resize, message_to_pass, path_name, path_file_name



def train_collate_covla_fn(batch):
  front_images =[torch.from_numpy(el[0]).permute(0,3,1,2) for el in batch]
  messages = [el[1] for el in batch]
  dirname = [el[2] for el in batch]
  filenames = [el[3] for el in batch]
  batch_process = [front_images, messages, dirname, filenames]
  return batch_process




"""
def device_to_camera(P_device, extrinsic_matrix):
    ""Convert device coordinates to camera coordinates.""
    P_device_hom = np.append(P_device, 1)
    P_camera_hom = np.dot(extrinsic_matrix, P_device_hom)
    return P_camera_hom[:3]

def camera_to_image(P_camera, intrinsic_matrix):
    ""Convert camera coordinates to image coordinates.""
    P_image_homogeneous = np.dot(intrinsic_matrix, P_camera)
    P_image = P_image_homogeneous[:2] / P_image_homogeneous[2]
    return P_image

def plot_trajectory_on_image(
    frame, trajectory, extrinsic_matrix, intrinsic_matrix, string_save ,marker="o", color="red"
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
    plt.figure()
    plt.imshow(frame)
    plt.axis("off")
    if len(future_pos_image) > 0:
        plt.plot(
            future_pos_image[:, 0], future_pos_image[:, 1],
            marker=marker, color=color, linestyle="solid", linewidth=1, markersize=3,
        )
    plt.savefig(string_save)






  plot_trajectory_on_image(
        frame=video_array[start_point-60],
        trajectory=states_array[start_point-60][str(start_point-60)]['trajectory'],
        extrinsic_matrix=np.array(states_array[start_point-60][str(start_point-60)]["extrinsic_matrix"]),
        intrinsic_matrix=np.array(states_array[start_point-60][str(start_point-60)]["intrinsic_matrix"]),
        string_save='/cluster/home/arsood/Semester_Project_Official/fig_3.png',
        marker="o",
        color="red"
        )

        plot_trajectory_on_image(
        frame=video_array[start_point],
        trajectory=transform_points,#states_array[start_point][str(start_point)]['trajectory'],
        extrinsic_matrix=np.array(states_array[start_point][str(start_point)]["extrinsic_matrix"]),
        intrinsic_matrix=np.array(states_array[start_point][str(start_point)]["intrinsic_matrix"]),
        string_save='/cluster/home/arsood/Semester_Project_Official/fig_4.png',
        marker="o",
        color="red"
        )

        plot_trajectory_on_image(
        frame=video_array[start_point+60],
        trajectory=states_array[start_point+60][str(start_point+60)]['trajectory'],
        extrinsic_matrix=np.array(states_array[start_point+60][str(start_point+60)]["extrinsic_matrix"]),
        intrinsic_matrix=np.array(states_array[start_point+60][str(start_point+60)]["intrinsic_matrix"]),
        string_save='/cluster/home/arsood/Semester_Project_Official/fig_5.png',
        marker="o",
        color="red"
        )
  
  
  """