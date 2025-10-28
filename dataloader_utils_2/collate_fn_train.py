import numpy as np
import torch

def train_collate(batch):
  front_image_list =[torch.from_numpy(el[0]).permute(0,3,1,2) for el in batch]
  front_images_list_no =[torch.from_numpy(el[0]) for el in batch]
  next_state_traj = [el[1][...,0:2] for el in batch]
  past_state_traj = [el[2][...,0:2] for el in batch]
  linear_path_vel = [el[3][...,0:2] for el in batch]
  num_intent = [el[4] for el in batch]
  cur_vel = [el[5] for el in batch]
  cur_acc = [el[6] for el in batch]

  batch_dict = {
        "front_image_list": front_image_list,
        "past_state_traj":past_state_traj,
        "next_state_traj": next_state_traj,
        "front_images_list_no" : front_images_list_no,
        "num_intent" : num_intent,
        "linear_path_vel" : linear_path_vel,
        "cur_vel": cur_vel,
        "cur_acc": cur_acc,
    }
  return batch_dict