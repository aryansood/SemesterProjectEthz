import cv2
import os
import imageio.v3 as iio
from tqdm import tqdm
import json
import torch
from model.qwen_model import QwenModel
import numpy as np
from dataloader_utils.dataloader_covla_train import CovlaDatasetTraining, train_collate_covla_fn
from torch.utils.data import DataLoader
#model = QwenModel(model_to_use=72, device='auto')
# training_dataset = WaymoE2EDatasetTraining('/cluster/scratch/arsood/data_clean', 4)
# training_loader = DataLoader(training_dataset, batch_size=1, shuffle=False, collate_fn=train_collate_fn)
#dir_to_save = '/cluster/scratch/arsood/data_strings_covla'

import matplotlib.pyplot as plt
import numpy as np

dir_path_videos = '/cluster/scratch/arsood/covla/videos'
dir_path_states  = '/cluster/scratch/arsood/covla/states'

make_dir = '/cluster/scratch/arsood/covla_string'

batch_size_n = 1
covla_dataset = CovlaDatasetTraining(dir_path_videos, dir_path_states, list(range(0,5000)))
covla_loader = DataLoader(covla_dataset, batch_size=batch_size_n, shuffle=True)#, collate_fn=train_collate_fn)

for batch in tqdm(covla_loader, desc="Loading batches"):

    
    l = input("contnue? ")


