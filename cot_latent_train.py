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
from model.qwen_cot_latent import QwenLatentCot


dataset_waymo = WaymoE2EDatasetTraining(data_waymo_train.data, 4)

dataset = ConcatDataset([dataset_waymo])
dataloader = DataLoader(dataset, batch_size = 2, shuffle=False, collate_fn=train_collate)

model_to_load = "/cluster/scratch/arsood/Qwen_2_5_vlm"
model_to_load_2 = "/cluster/scratch/arsood/Qwen_Fine_Tune_Waymo_Data_Final"
model = QwenLatentCot(model_to_load_2)

for batch in tqdm(dataloader):
    output = model(batch)