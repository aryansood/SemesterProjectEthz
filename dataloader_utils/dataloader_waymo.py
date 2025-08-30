import torch
from torch.utils.data import Dataset
import os
from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2
import random
import numpy as np
import tensorflow as tf
import cv2
from .strings import training_prompt
import json
import zipfile


def return_front3_cameras(data: wod_e2ed_pb2.E2EDFrame):
  image_list = []
  calibration_list = []
  order = [4,2,1,3,5]
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

def train_collate_fn(batch):
  front_images =[torch.from_numpy(el[0]).permute(0,3,1,2) for el in batch]
  rear_image =[torch.from_numpy(el[1]).permute(0,3,1,2) for el in batch]
  next_state_traj = [el[2][..., 0:2] for el in batch]
  past_state_traj = [el[3][..., 0:2] for el in batch]
  intent_traj = [el[4] for el in batch]
  messages = [el[5] for el in batch]
  
  batch_process = [front_images, rear_image, next_state_traj, past_state_traj, intent_traj, messages]
  return batch_process

def return_objects(interval_start, interval_end, file_data_path, file_data_names):

  past_state_traj = None
  next_state_traj = None
  front_image_list = []
  rear_image_list = []
  driving_intent = ""
  resize_factor = 5

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
          
          front3_camera_image_list, front3_camera_calibration_list = return_front3_cameras(data)  
          front_concatenated = np.concatenate(front3_camera_image_list, axis=1)
          front_concatenated = cv2.resize(front_concatenated, (int(front_concatenated.shape[1]/resize_factor), int(front_concatenated.shape[0]/resize_factor)), interpolation=cv2.INTER_AREA) 
          front_image_list.append(front_concatenated)
          
          rear3_camera_image_list, rear3_camera_calibration_list = return_rear3_cameras(data)
          rear3_camera_image_list[1] = cv2.resize(rear3_camera_image_list[1], (rear3_camera_image_list[0].shape[1], rear3_camera_image_list[0].shape[0]))
          rear_concatenated = np.concatenate(rear3_camera_image_list, axis=1)
          rear_concatenated = cv2.resize(rear_concatenated, (int(rear_concatenated.shape[1]/resize_factor), int(rear_concatenated.shape[0]/resize_factor)), interpolation=cv2.INTER_AREA) 
          rear_image_list.append(rear_concatenated)
    
  
  front_image_list = np.array(front_image_list)
  rear_image_list = np.array(rear_image_list)
  next_state_traj = np.array(next_state_traj)
  past_state_traj = np.array(past_state_traj)

  return front_image_list, rear_image_list, next_state_traj, past_state_traj, driving_intent
   


class WaymoE2EDatasetTrainingAnnotated(Dataset):
    def __init__(self, data_path, seq_len, is_fut_traj = True):
        super().__init__()

        self.data_path = data_path
        self.zip_annot_path = "data/waymo_annot.zip"

        self.zip_ref_annot = zipfile.ZipFile(self.zip_annot_path, 'r')
        self.list_annotated_files = [f for f in self.zip_ref_annot.namelist() if not f.endswith('/')]

        self.dir_list = os.listdir(data_path)
        
        self.seq_len = seq_len
        self.is_fut_traj = is_fut_traj

    def __len__(self):
        return len(self.list_annotated_files)
    
    def __getitem__(self, index):
        
        file_data_path_name = os.path.splitext(self.list_annotated_files[index])[0]
        video_idf = os.path.basename(os.path.dirname(file_data_path_name))
        video_data_path = os.path.join(self.data_path, video_idf)
        file_data_names = sorted(os.listdir(os.path.join(self.data_path, video_idf)))
        filename_index_name = f"{os.path.basename(file_data_path_name)}.bin"
        interval = file_data_names.index(filename_index_name)
        front_image_list, rear_image_list, next_state_traj, past_state_traj, drving_intent = return_objects(interval-self.seq_len+1, interval+1, video_data_path, file_data_names)

        annotated_data = ""
        with self.zip_ref_annot.open(self.list_annotated_files[index]) as file_annot:
           annotated_data = np.load(file_annot, allow_pickle=True)

        np.set_printoptions(precision=4, suppress=True)
        prompt_to_use = training_prompt(np.array_str(past_state_traj), drving_intent, np.array_str(next_state_traj))
        indices = [0, 3, 7, 11, 15, 19]
        next_state_traj_5 = next_state_traj[indices]

        clean_string = str(annotated_data).strip()
        if clean_string.startswith("```json"):
            clean_string = clean_string[len("```json"):].strip()
        if clean_string.endswith("```"):
            clean_string = clean_string[:-3].strip()

        #print(annotated_data_ext)
        gt_label = json.loads(str(clean_string))
        
        if(self.is_fut_traj):
           gt_label["traj_fut"] = next_state_traj_5[..., 0:2].tolist()
        gt_label = json.dumps(gt_label)

        message_to_pass = [{
        "role": "user",
        "content": [
            {"type": "video", "video": ""},
            {"type": "text", "text": prompt_to_use},
        ],
        }
        ,{
            "role": "assistant",
            "content": [{"type": "text", "text": gt_label}],
        }
        ]

        return front_image_list, rear_image_list, next_state_traj, past_state_traj, drving_intent, message_to_pass
    
class WaymoE2EDatasetTraining(Dataset):
    def __init__(self, data_path, seq_len):
        super().__init__()
        self.data_path = data_path
        self.dir_list = os.listdir(data_path)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.dir_list)
    
    def __getitem__(self, index):
        
        file_data_path = os.path.join(self.data_path, self.dir_list[index])
        file_data_names = sorted(os.listdir(file_data_path))

        interval = random.randint(0, len(file_data_names)-self.seq_len)

        front_image_list, rear_image_list, next_state_traj, past_state_traj, drving_intent = return_objects(interval, interval+self.seq_len, file_data_path, file_data_names)

        np.set_printoptions(precision=4, suppress=True)

        prompt_to_use = training_prompt(np.array_str(past_state_traj), drving_intent, np.array_str(next_state_traj))

        indices = [0, 3, 7, 11, 15, 19]
        next_state_traj_5 = next_state_traj[indices]

        message_to_pass = [{
        "role": "user",
        "content": [
            {"type": "video", "video": ""},
            {"type": "text", "text": prompt_to_use },
        ],
        }
        ,{
            "role": "assistant",
            "content": [{"type": "text", "text": "{"+np.array_str(next_state_traj_5[...,:2])+"}"}],
        }
        ]

        return front_image_list, rear_image_list, next_state_traj, past_state_traj, drving_intent, message_to_pass
    
class WaymoE2EDatasetLabeler(Dataset):
    def __init__(self, data_path, seq_len, self_cur_idx = 10):
        super().__init__()
        self.data_path = data_path
        self.dir_list = os.listdir(data_path)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.dir_list)
    
    def __getitem__(self, index):
        
        file_data_path = os.path.join(self.data_path, self.dir_list[index])
        file_data_names = sorted(os.listdir(file_data_path))

        interval = self.self_cur_idx*10
        interval = random.randint(interval, interval+10-self.seq_len)
        if(interval >= len(file_data_names)-10):
           interval = len(file_data_names)-self.seq_len-10

        front_image_list, rear_image_list, next_state_traj, past_state_traj, drving_intent = return_objects(interval, interval+self.seq_len, file_data_path, file_data_names)

        np.set_printoptions(precision=4, suppress=True)

        prompt_to_use = training_prompt(np.array_str(past_state_traj), drving_intent, np.array_str(next_state_traj))

        indices = [0, 3, 7, 11, 15, 19]
        #next_state_traj_5 = next_state_traj[indices]

        message_to_pass = [{
        "role": "user",
        "content": [
            {"type": "video", "video": ""},
            {"type": "text", "text": prompt_to_use },
        ],
        }
        ]

        return front_image_list, rear_image_list, next_state_traj, past_state_traj, drving_intent, message_to_pass

