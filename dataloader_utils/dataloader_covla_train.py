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
    def __init__(self, data_path_videos, data_path_states, indices, seq_len = 4, self_cur_idx = 5):
        super().__init__()
        self.data_path_videos = data_path_videos
        self.list_videos = os.listdir(self.data_path_videos)
        self.list_videos = sorted(self.list_videos)

        self.data_path_states = data_path_states
        self.list_states = os.listdir(self.data_path_states)
        self.list_states = sorted(self.list_states)

        self.list_videos = [self.list_videos[idx] for idx in indices]
        self.list_states = [self.list_states[idx] for idx in indices]

        self.window = 60
        self.seq_len = seq_len
        self.self_cur_idx = self_cur_idx

    def __len__(self):
        return len(self.list_states)
    
    def __getitem__(self, index):
        file_path_video = os.path.join(self.data_path_videos, self.list_videos[index])
        file_path_states = os.path.join(self.data_path_states, self.list_states[index])

        interval = self.window*self.self_cur_idx
        candidate_interval = list(range(interval, interval+60-self.seq_len, 3))
        start_point = random.choice(candidate_interval)

        video_array = iio.imread(file_path_video, index=None)
        states_array = []
        conta = []
        l_conta = 0
        with open(file_path_states, 'r') as f:
            for line in f:
                if line.strip():
                    states_array.append(json.loads(line))
                    if(states_array[-1][str(l_conta)]['trajectory_count'] != 60):
                        conta.append(l_conta)
                    l_conta+=1
        video_array = video_array[start_point:start_point+self.seq_len]

        trajectory = np.array(states_array[start_point][str(start_point)]['trajectory'])

        # l2 = np.array(states_array[start_point+40][str(start_point+40)]['orientations_ned'])
        # l1 = np.array(states_array[start_point][str(start_point)]['orientations_ned'])
        # #angle_rad = l2[-1]-l1[-1]

        traj_present = np.array(states_array[start_point][str(start_point)]['trajectory'])[-2:, 0:2]
        traj_future = np.array(states_array[start_point+60][str(start_point+60)]['trajectory'])[0:40,0:2]
        traj_past = np.array(states_array[start_point-60][str(start_point-60)]['trajectory'])[0:40,0:2]

        angle_past_rad = math.atan((traj_past[-1][1]-traj_past[-2][1])/(traj_past[-1][0]-traj_past[-2][0]))
        rot_mat_past = np.array([
            [math.cos(angle_past_rad),-math.sin(angle_past_rad)],
            [math.sin(angle_past_rad),math.cos(angle_past_rad)]
        ])
        traj_past = traj_past-traj_past[-1]
        transform_points_past = np.matmul(traj_past, rot_mat_past)

        angle_rad = math.atan((traj_present[1][1]-traj_present[0][1])/(traj_present[1][0]-traj_present[0][0]))
        rot_mat = np.array([
            [math.cos(angle_rad),-math.sin(angle_rad)],
            [math.sin(angle_rad),math.cos(angle_rad)]
        ])
        print(traj_future.shape)
        print(rot_mat.shape)
        transform_points = np.matmul(traj_future, rot_mat.T)
        transform_points = np.array(states_array[start_point][str(start_point)]['trajectory'])[-1, 0:2]+transform_points
        plt.figure()
        plt.xlim([-20,20])
        plt.plot(trajectory[...,1], trajectory[...,0], color='red')
        plt.plot(transform_points[...,1], transform_points[...,0], color='blue')
        plt.plot(transform_points_past[..., 1], transform_points_past[..., 0], color='green')
        plt.savefig('/cluster/home/arsood/Semester_Project_Official/fig_2.png')

        return video_array, trajectory



def train_collate_covla_fn(batch):
  front_images =[torch.from_numpy(el[0]).permute(0,3,1,2) for el in batch]
  intent_traj = [el[4] for el in batch]
  messages = [el[5] for el in batch]
  filenames = [el[6] for el in batch]
  dir_name = [el[7] for el in batch]
  batch_process = [front_images, intent_traj, messages, filenames, dir_name]
  return batch_process