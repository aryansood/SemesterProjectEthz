from dataloader_utils_2.waymo_dataloader_split import WaymoE2EDatasetTraining
from dataloader_utils_2.covla_dataloader_split import CovlaDatasetTraining
from dataloader_utils_2.collate_fn_train import train_collate
from dataloader_utils_2.waymo_val_dataloader import WaymoE2EDatasetVal
from torch.utils.data import DataLoader, ConcatDataset
from config import data_waymo_train, data_covla, data_waymo_val
from tqdm import tqdm
import numpy as np
import torch
import os
from train.train_distributed import trainer
from torch.utils.data import random_split
from config import data_waymo_train
from video_utils.video_utils import save_to_mp4
# from model.dino_autoregr_decoder import PipiLineDinoAutoregr
from model.dino_adjust_decoder import PipiLineDinoAdjust
import matplotlib.pyplot as plt

dataset_waymo = WaymoE2EDatasetTraining(data_waymo_train.data, 1)
dataset_covla = CovlaDatasetTraining(data_covla.video, data_covla.state, 1)

dataset = ConcatDataset([dataset_covla])

train_dataset = dataset
val_dataset = WaymoE2EDatasetVal(data_waymo_val.data, 1)

# dataset_size = len(dataset)
# val_size = int(0.1 * dataset_size)
# train_size = dataset_size - val_size
#train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

load_dest_path = "/cluster/scratch/arsood/Diritto_Val/model.pth"
save_val_dest_path = "/cluster/scratch/arsood/Diritto_Val_1/model.pth"
save_final_dest_path = "/cluster/scratch/arsood/Diritto_Train/model.pth"
tensor_board_path = "/cluster/scratch/arsood/Diritto_Tensor_4"


model = PipiLineDinoAdjust()

#model.load_state_dict(torch.load(save_val_dest_path, weights_only=True), strict=False)

trainer(model, train_dataset, 10, 30, train_collate, val_dataset=val_dataset, val_steps= 100, save_val_dest_path = save_val_dest_path, save_final_dest_path = save_final_dest_path, tensor_board_path = tensor_board_path, max_val_step = 24)





