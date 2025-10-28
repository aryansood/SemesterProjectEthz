import torch
from torch.utils.data import Dataset
import os
import random
import numpy as np
import cv2
import json
import random
import math
import matplotlib.pyplot as plt
import zipfile
import av
from decord import VideoReader, cpu
from collections import defaultdict

def return_objects(start_point, file_path_video, file_path_states, seq_len):
    resize_factor = 4
    #video_array = iio.imread(file_path_video, index=None)

    states_array = []
    driving_intent = "GO_STRAIGHT"
    num_intent = 1
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

    vr = VideoReader(file_path_video, ctx=cpu(0))
    frame_indices = range(start_point-(seq_len*2-1), start_point+1)
    video_array_1 = vr.get_batch(frame_indices).asnumpy()
    video_array_1 = (video_array_1[::-2])[::-1]
    video_array_resize = [cv2.resize(video_array_1[j], (int(video_array_1[j].shape[1]/resize_factor), int(video_array_1[j].shape[0]/resize_factor)), interpolation=cv2.INTER_AREA) for j in range(0,video_array_1.shape[0])]
    video_array_resize = np.array(video_array_resize)
    
    if(states_array[start_point][str(start_point)]['rightBlinker'] == True):
        driving_intent = 'GO_RIGHT'
        num_intent = 3
    elif (states_array[start_point][str(start_point)]['leftBlinker'] == True):
        driving_intent = 'GO_LEFT'
        num_intent = 2
    
    cur_vel = np.array(states_array[start_point][str(start_point)]['velocities_calib'])[0:2]
    cur_acc = np.array(states_array[start_point][str(start_point)]['accelerations_calib'])[0:2]
    
    return video_array_resize, driving_intent, transform_points_past, transform_points, num_intent, cur_vel, cur_acc
    

class CovlaDatasetTraining(Dataset):
    def __init__(self, data_path_videos, data_path_states, seq_len, split_ratio = [0.05, 0.5, 1, 1]):
        super().__init__()
        self.data_path_videos = data_path_videos
        self.list_videos = os.listdir(self.data_path_videos)
        self.list_videos = sorted(self.list_videos)

        self.data_path_states = data_path_states
        self.list_states = os.listdir(self.data_path_states)
        self.list_states = sorted(self.list_states)

        self.seq_len = seq_len

        covla_split_train = np.load("dataloader_utils_2/train_data_split/covla_disjoint_set.npy", allow_pickle=True)

        covla_split_0_1 = random.sample(covla_split_train[0], int(len(covla_split_train[0]) * split_ratio[0]))
        covla_split_1_2 = random.sample(covla_split_train[1], int(len(covla_split_train[1]) * split_ratio[1]))
        covla_split_2_3 = random.sample(covla_split_train[2], int(len(covla_split_train[2]) * split_ratio[2]))
        covla_split_3_4 = random.sample(covla_split_train[3], int(len(covla_split_train[3]) * split_ratio[3]))

        self.covla_data_set = covla_split_0_1 + covla_split_1_2 + covla_split_2_3 + covla_split_3_4

        covla_grouped = defaultdict(list)
        for key, value in self.covla_data_set:
             covla_grouped[key].append(value)
        self.covla_data_set = sorted(list(covla_grouped.items()))

    def __len__(self):
        return len(self.covla_data_set)
    
    def __getitem__(self, index):
        file_path_video = os.path.join(self.data_path_videos,f"{self.covla_data_set[index][0].split('.')[0]}.mp4")
        file_path_states = os.path.join(self.data_path_states, f"{self.covla_data_set[index][0].split('.')[0]}.jsonl")

        start_point = random.choice(self.covla_data_set[index][1])

        front_image_list, driving_intent, past_state_traj, next_state_traj, num_intent, cur_vel, cur_acc = return_objects(start_point, file_path_video, file_path_states, self.seq_len)

        list_time_step = np.arange(0.25 , 5.25, 0.25)
        linear_path_vel_x = (max(0, cur_vel[0])*list_time_step)[:, None]
        linear_path_vel_y = (0*list_time_step)[:, None]
        linear_path_concat = np.concatenate([linear_path_vel_x, linear_path_vel_y], axis = -1)
        return front_image_list, next_state_traj, past_state_traj, linear_path_concat, num_intent, cur_vel, cur_acc