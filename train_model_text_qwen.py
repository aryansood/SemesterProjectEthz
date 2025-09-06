from dataloader_utils.dataloader_waymo import WaymoE2EDatasetTrainingAnnotated
from dataloader_utils.dataloader_covla import CovlaDatasetTrainingAnnotated, CovlaDatasetTraining
from dataloader_utils.dataloader_val_waymo import WaymoE2EDatasetVal, val_collate_waymo
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

model = QwenFineTunedModelText()
training_covla_annotated = CovlaDatasetTrainingAnnotated(data_covla.video, data_covla.state, 4, is_fut_traj=False)
training_waymo_annotated = WaymoE2EDatasetTrainingAnnotated(data_waymo_train.data, 4, is_fut_traj=False)
training_data = ConcatDataset([training_covla_annotated, training_waymo_annotated])

dataset_size = len(training_data)
val_size = int(0.2 * dataset_size)
train_size = dataset_size - val_size

torch.manual_seed(42)
train_dataset, val_dataset = random_split(training_data, [train_size, val_size])


trainer(model, train_dataset, 2, 2, train_collate, val_dataset=val_dataset, val_steps= 100)

