import numpy as np
import torch

def train_collate(batch):
  front_images =[torch.from_numpy(el[0]).permute(0,3,1,2) for el in batch]
  front_images_no =[torch.from_numpy(el[0]) for el in batch]
  messages = [el[1] for el in batch]
  past_traj = [el[2] for el in batch]
  fut_traj = [el[3] for el in batch]
  batch_dict = {
        "messages": messages,
        "front_images": front_images,
        "past_state_traj": past_traj,
        "next_state_traj": fut_traj,
        "front_images_no" : front_images_no,
    }
  return batch_dict