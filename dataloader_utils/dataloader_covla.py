import torch
from torch.utils.data import Dataset
import os
import random
import numpy as np
import cv2
import json
from .prompt_train import training_prompt_covla, training_prompt_covla_direct_traj
import random
import math
import matplotlib.pyplot as plt
import zipfile
import av
from decord import VideoReader, cpu

def return_objects(start_point, file_path_video, file_path_states):
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
    frame_indices = range(start_point-7, start_point+1)
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

class CovlaDatasetTrainingAnnotated(Dataset):
    def __init__(self, data_path_videos, data_path_states, seq_len = 4, is_fut_traj = False):
        super().__init__()
        self.data_path_videos = data_path_videos
        self.list_videos = os.listdir(self.data_path_videos)
        self.list_videos = sorted(self.list_videos)

        self.data_path_states = data_path_states
        self.list_states = os.listdir(self.data_path_states)
        self.list_states = sorted(self.list_states)

        self.zip_annot_path = "data/covla_annot.zip"
        self.zip_ref_annot = zipfile.ZipFile(self.zip_annot_path, 'r')
        self.list_annotated_files = [f for f in self.zip_ref_annot.namelist() if not f.endswith('/')]

        self.seq_len = seq_len
        self.is_fut_traj = is_fut_traj

    def __len__(self):
        return len(self.list_annotated_files)
    
    def __getitem__(self, index):
        file_data_path_name = os.path.splitext(self.list_annotated_files[index])[0]
        drive_seg_idf = os.path.basename(os.path.dirname(file_data_path_name))#file_data_path_name
        drive_seg_idf = drive_seg_idf.split("_")[0]

        video_data_path = f"{os.path.join(self.data_path_videos, drive_seg_idf)}.mp4"
        state_data_path = f"{os.path.join(self.data_path_states, drive_seg_idf)}.jsonl"

        start_point = int(self.list_annotated_files[index].split("-")[-1].split(".")[0])#int(self.list_annotated_files[index].split("-")[-1].split("_")[-1].split(".")[0])#
        
        front_image_list, driving_intent, past_state_traj, next_state_traj, num_intent, cur_vel, cur_acc = return_objects(start_point, video_data_path, state_data_path)

        annotated_data = ""
        with self.zip_ref_annot.open(self.list_annotated_files[index]) as file_annot:
           annotated_data = np.load(file_annot, allow_pickle=True)

        indices = [0, 3, 7, 11, 15, 19]
        next_state_traj_5 = next_state_traj[indices]

        clean_string = str(annotated_data).strip()
        if clean_string.startswith("```json"):
            clean_string = clean_string[len("```json"):].strip()
        if clean_string.endswith("```"):
            clean_string = clean_string[:-3].strip()
        
        np.set_printoptions(suppress=True)
        gt_label = json.loads(str(clean_string))
        if(self.is_fut_traj):
           gt_label["traj_fut"] = np.round(next_state_traj_5[..., 0:2], 2).tolist()
        gt_label = json.dumps(gt_label)
        
        prompt_to_use = training_prompt_covla(driving_intent, np.array_str(np.round(past_state_traj[..., 0:2], 1)), np.array_str(np.round(cur_vel, 4)), np.array_str(np.round(cur_acc, 4)))
        message_to_pass = [{
        "role": "user",
        "content": [
            {"type": "video", "video": ""},
            {"type": "text", "text": prompt_to_use },
        ],
        }
        ,{
            "role": "assistant",
            "content": [{"type": "text", "text": gt_label}],
        }
        ]
        print(cur_vel) 
        print(cur_acc)
        return front_image_list, message_to_pass, past_state_traj, next_state_traj, num_intent, num_intent
    

class CovlaDatasetTraining(Dataset):
    def __init__(self, data_path_videos, data_path_states, seq_len = 4, indices = list(range(0, 9999))):
        super().__init__()
        self.data_path_videos = data_path_videos
        self.list_videos = os.listdir(self.data_path_videos)
        self.list_videos = sorted(self.list_videos)

        self.data_path_states = data_path_states
        self.list_states = os.listdir(self.data_path_states)
        self.list_states = sorted(self.list_states)

        self.list_videos = [self.list_videos[idx] for idx in indices]
        self.list_states = [self.list_states[idx] for idx in indices]

        self.seq_len = seq_len

    def __len__(self):
        return len(self.list_states)
    
    def __getitem__(self, index):
        file_path_video = os.path.join(self.data_path_videos, self.list_videos[index])
        file_path_states = os.path.join(self.data_path_states, self.list_states[index])

        start_point = random.randint(80,460)

        front_image_list, driving_intent, past_state_traj, next_state_traj, num_intent, cur_vel, cur_acc = return_objects(start_point, file_path_video, file_path_states)
        
        indices = [0, 3, 7, 11, 15, 19]
        next_state_traj_5 = next_state_traj[indices]
        np.set_printoptions(suppress=True)

        gt_label = {"traj_fut": np.round(next_state_traj_5[..., 0:2], 2).tolist()}
        gt_label = json.dumps(gt_label)

        prompt_to_use = training_prompt_covla(driving_intent, np.array_str(np.round(past_state_traj[..., 0:2], 1)), np.array_str(np.round(cur_vel, 4)), np.array_str(np.round(cur_acc, 4)))

        message_to_pass = [{
        "role": "user",
        "content": [
            {"type": "video", "video": ""},
            {"type": "text", "text": prompt_to_use },
        ],
        }
        # ,{
        #     "role": "assistant",
        #     "content": [{"type": "text", "text": gt_label}],
        # }
        ]
        name, ext = os.path.splitext(self.list_videos[index])
        name = f"{name}_{start_point}"
        return front_image_list, message_to_pass, past_state_traj, next_state_traj, num_intent, num_intent



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





"""
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
        )"""