import numpy as np
import torch

def train_collate(batch):
  front_image_list =[torch.from_numpy(el[0]).permute(0,3,1,2) for el in batch]
  front_images_list_no =[torch.from_numpy(el[0]) for el in batch]
  next_state_traj = [el[1][...,0:2] for el in batch]
  past_state_traj = [el[2][...,0:2] for el in batch]
  messages =  [el[3] for el in batch]
  batch_dict = {
        "front_images": front_image_list,
        "past_state_traj":past_state_traj,
        "next_state_traj": next_state_traj,
        "front_images_list_no" : front_images_list_no,
        "messages" : messages,
    }
  return batch_dict