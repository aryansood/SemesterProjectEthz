from dataloader_utils_2.covla_dataloader_split import CovlaDatasetTraining
from dataloader_utils_2.waymo_dataloader_split import WaymoE2EDatasetTraining
from dataloader_utils_2.collate_fn_train import train_collate
from torch.utils.data import DataLoader, ConcatDataset
from config import data_covla, data_waymo_train
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
from tqdm import tqdm
from dataloader_utils.dataloader_val_waymo import WaymoE2EDatasetVal, collate_val
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor

# training_covla_annotated = CovlaDatasetTraining(data_covla.video, data_covla.state, 4)
training_waymo = WaymoE2EDatasetTraining(data_waymo_train.data, 4)
# training_data = training_covla_annotated

# dataset_size = len(training_data)
# val_size = int(0.1 * dataset_size)
# train_size = dataset_size - val_size

train_dataset = training_waymo#random_split(training_data, [train_size, val_size])

save_val_dest_path = '/cluster/scratch/arsood/qwen_fine_tune_covla_wcot_val_2'
save_final_dest_path = '/cluster/scratch/arsood/qwen_fine_tune_covla_wcot_final_2'
tensor_board_path = '/cluster/scratch/arsood/qwen_fine_tune_covla_wcot_board_2'

prcoessor_config = "/cluster/scratch/arsood/Qwen_2_5_vlm"
model_to_load = "/cluster/scratch/arsood/Qwen_2_5_vlm"

model = QwenFineTunedModelText(save_final_dest_path, prcoessor_config)
#trainer(model, train_dataset, 2, 2, train_collate, val_dataset=None, val_steps= 100, save_val_dest_path = save_val_dest_path, save_final_dest_path = save_final_dest_path, tensor_board_path = tensor_board_path, max_val_step=30)
dataloader_waymo = DataLoader(training_waymo, batch_size=20, collate_fn=train_collate)
for batch in tqdm(dataloader_waymo):
    l = 10
    output = model.generate(batch)
    for el in output:
        print(el)