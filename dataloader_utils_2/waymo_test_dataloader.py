import torch
from torch.utils.data import Dataset
import os
from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2
import random
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict
from .prompt_train import training_prompt_waymo, training_prompt_waymo_direct_traj
import json

def return_front3_cameras(data: wod_e2ed_pb2.E2EDFrame):
  image_list = []
  calibration_list = []
  order = [2,1,3]
  for camera_name in order:
    for index, image_content in enumerate(data.frame.images):
      if image_content.name == camera_name:
        calibration = data.frame.context.camera_calibrations[index]
        image = tf.io.decode_image(image_content.image).numpy()
        image_list.append(image)
        calibration_list.append(calibration)
        break

  return image_list, calibration_list

def return_rear3_cameras(data: wod_e2ed_pb2.E2EDFrame):
  image_list = []
  calibration_list = []
  order = [8,7,6]
  for camera_name in order:
    for index, image_content in enumerate(data.frame.images):
      if image_content.name == camera_name:
        calibration = data.frame.context.camera_calibrations[index]
        image = tf.io.decode_image(image_content.image).numpy()
        image_list.append(image)
        calibration_list.append(calibration)
        break

  return image_list, calibration_list

def test_collate(batch):
  front_image_list =[torch.from_numpy(el[0]).permute(0,3,1,2) for el in batch]
  front_images_list_no =[torch.from_numpy(el[0]) for el in batch]
  next_state_traj = [el[1][...,0:2] for el in batch]
  past_state_traj = [el[2][...,0:2] for el in batch]
  messages =  [el[3] for el in batch]
  frame_names = [el[4] for el in batch]
  batch_dict = {
        "front_images": front_image_list,
        "past_state_traj":past_state_traj,
        "next_state_traj": next_state_traj,
        "front_images_list_no" : front_images_list_no,
        "messages" : messages,
        "frame_names" : frame_names,
    }
  return batch_dict

def return_objects(interval_start, interval_end, file_data_path, file_data_names):

  past_state_traj = None
  next_state_traj = None
  cur_vel =  None
  cur_acc = None
  front_image_list = []
  rear_image_list = []
  driving_intent = ""
  resize_factor = 5
  num_intent = None
  frame_name = None

  direction_dist = {
     0: "UNKNOWN",
    1: "GO_STRAIGHT",
    2: "GO_LEFT",
    3: "GO_RIGHT"
  }

  for el in range(interval_start, interval_end):
      filepath = os.path.join(file_data_path, file_data_names[el])
      with open(filepath, 'rb') as file:
          data_bin = file.read()
          data = wod_e2ed_pb2.E2EDFrame()
          data.ParseFromString(data_bin)
          next_state_traj = np.stack([data.future_states.pos_x, data.future_states.pos_y, np.zeros_like(data.future_states.pos_x)], axis=1)
          past_state_traj = np.stack([data.past_states.pos_x, data.past_states.pos_y, np.zeros_like(data.past_states.pos_x)], axis=1)
          
          driving_intent = direction_dist[data.intent]
          num_intent = data.intent
          
          front3_camera_image_list, front3_camera_calibration_list = return_front3_cameras(data)  
          front_concatenated = np.concatenate(front3_camera_image_list, axis=1)
          front_concatenated = cv2.resize(front_concatenated, (int(front_concatenated.shape[1]/resize_factor), int(front_concatenated.shape[0]/resize_factor)), interpolation=cv2.INTER_AREA) 
          front_image_list.append(front_concatenated)
          
          rear3_camera_image_list, rear3_camera_calibration_list = return_rear3_cameras(data)
          rear3_camera_image_list[1] = cv2.resize(rear3_camera_image_list[1], (rear3_camera_image_list[0].shape[1], rear3_camera_image_list[0].shape[0]))
          rear_concatenated = np.concatenate(rear3_camera_image_list, axis=1)
          rear_concatenated = cv2.resize(rear_concatenated, (int(rear_concatenated.shape[1]/resize_factor), int(rear_concatenated.shape[0]/resize_factor)), interpolation=cv2.INTER_AREA) 
          rear_image_list.append(rear_concatenated)

          cur_vel = np.stack([data.past_states.vel_x[-1], data.past_states.vel_y[-1]])
          cur_acc = np.stack([data.past_states.accel_x[-1], data.past_states.accel_y[-1]])
          frame_name = data.frame.context.name

          
    
  
  front_image_list = np.array(front_image_list)
  rear_image_list = np.array(rear_image_list)
  next_state_traj = np.array(next_state_traj)
  past_state_traj = np.array(past_state_traj)

  return front_image_list, rear_image_list, next_state_traj, past_state_traj, driving_intent, num_intent, cur_vel, cur_acc, frame_name
   

class WaymoE2EDatasetTest(Dataset):
    def __init__(self, data_path, seq_len):
        super().__init__()
        self.data_path = data_path
        self.dir_list = os.listdir(data_path)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.dir_list)
    
    def __getitem__(self, index):
        dir_loc = os.path.join(self.data_path, self.dir_list[index])
        file_names = os.listdir(dir_loc)
        file_names = sorted(file_names)
        interval = len(file_names)-1

        front_image_list, rear_image_list, next_state_traj, past_state_traj, driving_intent, num_intent, cur_vel, cur_acc, frame_name = return_objects(interval-self.seq_len, interval+1, dir_loc, file_names)
        
        np.set_printoptions(suppress=True)
        prompt_to_use = training_prompt_waymo_direct_traj(driving_intent, np.array_str(np.round(past_state_traj[..., 0:2], 1)), str(np.round(cur_vel[0], 4)), str(np.round(cur_acc[0], 4)))
        message_to_pass = [{
        "role": "user",
        "content": [
            {"type": "video", "video": ""},
            {"type": "text", "text": prompt_to_use },
        ],
        }
        ]
        return front_image_list, next_state_traj, past_state_traj, message_to_pass, frame_name