import cv2
import os
import imageio.v3 as iio
from tqdm import tqdm
import json
import torch
from model.qwen_model import QwenModel
import numpy as np

#model = QwenModel(model_to_use=72, device='auto')
# training_dataset = WaymoE2EDatasetTraining('/cluster/scratch/arsood/data_clean', 4)
# training_loader = DataLoader(training_dataset, batch_size=1, shuffle=False, collate_fn=train_collate_fn)
#dir_to_save = '/cluster/scratch/arsood/data_strings_covla'




dir_path_videos = '/cluster/scratch/arsood/covla/videos'
dir_path_states  = '/cluster/scratch/arsood/covla/states'
filenames_states = os.listdir(dir_path_states)
filename_videos = os.listdir(dir_path_videos)
filenames_states = sorted(filenames_states)
filename_videos = sorted(filename_videos)

make_dir = '/cluster/scratch/arsood/covla_string'
