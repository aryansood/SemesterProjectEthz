from dataloader_utils.dataloader_waymo import WaymoE2EDatasetTrainingAnnotated, WaymoE2EDatasetTraining
from dataloader_utils.dataloader_covla import CovlaDatasetTrainingAnnotated, CovlaDatasetTraining
from dataloader_utils.dataloader_val_waymo import WaymoE2EDatasetVal
from dataloader_utils.collate_functions import train_collate
from torch.utils.data import DataLoader, ConcatDataset
from config import data_waymo_train, data_covla, data_waymo_val
from tqdm import tqdm
import numpy as np
import torch
import os
from train.train_distributed import trainer
from model.qwen_finetuned_text import QwenFineTunedModelText
from torch.utils.data import random_split
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
import matplotlib.pyplot as plt
import statistics

val_dataset = WaymoE2EDatasetVal(data_waymo_val.data, 4)
# val_loader = DataLoader(val_dataset, batch_size=30, shuffle = True, collate_fn = train_collate)
load_path_model = '/cluster/scratch/arsood/qwen_with_fut_traj_val_2'
model = QwenFineTunedModelText(cache_model = load_path_model)

# for batch in tqdm(val_loader):
#     out = model_first.generate(batch)


training_covla_annotated = CovlaDatasetTrainingAnnotated(data_covla.video, data_covla.state, 4, is_fut_traj=True)
training_waymo_annotated = WaymoE2EDatasetTrainingAnnotated(data_waymo_train.data, 4, is_fut_traj=True)
training_data = ConcatDataset([training_covla_annotated, training_waymo_annotated])


# dataset_size = len(training_data)
# val_size = int(0.2 * dataset_size)
# train_size = dataset_size - val_size

# train_dataset, val_dataset = random_split(training_data, [train_size, val_size])

save_val_dest_path = '/cluster/scratch/arsood/qwen_with_fut_traj_val_3B_2'
save_final_dest_path = '/cluster/scratch/arsood/qwen_with_fut_traj_final_3B_2'
tensor_board_path = '/cluster/scratch/arsood/tensor_board_fut_traj_3B_2'
# model = QwenFineTunedModelText(cache_model= )
trainer(model, training_data, 2, 2, train_collate, val_dataset=val_dataset, val_steps= 500, save_val_dest_path = save_val_dest_path, save_final_dest_path = save_final_dest_path, tensor_board_path = tensor_board_path)

# l = 0
# sum_3 = 0
# sum_5 = 0
# list_3 = []
# list_5 = []
# for batch in tqdm(val_loader):
#     loss_avg = model_first.validate_traj_generate(batch)
#     ade_3_second = torch.mean(loss_avg[...,0:12].clone()).item()
#     ade_5_second = torch.mean(loss_avg).item()
#     list_3.append(ade_3_second)
#     list_5.append(ade_5_second)
#     print("Ade3: ", statistics.mean(list_3))
#     print("Ade5: ", statistics.mean(list_5))
