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
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
import matplotlib.pyplot as plt
model = QwenFineTunedModelText(cache_model = '/cluster/scratch/arsood/qwen_waymo_all_finetune_final_5000')
# training_covla_annotated = CovlaDatasetTrainingAnnotated(data_covla.video, data_covla.state, 4, is_fut_traj=False)
# training_waymo_annotated = WaymoE2EDatasetTrainingAnnotated(data_waymo_train.data, 4, is_fut_traj=False)
# training_data = ConcatDataset([training_covla_annotated, training_waymo_annotated])

# dataset_size = len(training_data)
# val_size = len(training_data)-20#int(0.2 * dataset_size)
# train_size = dataset_size - val_size

# torch.manual_seed(42)
# train_dataset, val_dataset = random_split(training_data, [train_size, val_size])

val_dataset = WaymoE2EDatasetVal(data_waymo_val.data, 4)
val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=train_collate, shuffle=True)

#optimizer = torch.optim.AdamW(model.peft_model.parameters(), lr=1e-5, weight_decay=0.0)

for batch in tqdm(val_loader):
    output = model.generate(batch)
    print(output[0])
    frames = []
    fig = plt.figure()
    for i in range(0, batch["front_images_no"][0].shape[0]):
        frames.append([plt.imshow(batch["front_images_no"][0][i], animated=True)])
    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)
    writer = FFMpegWriter(fps=20, bitrate=1800)
    ani.save('/cluster/home/arsood/Semester_Project_Official/video.mp4', writer=writer, dpi=300)
    input_l = input("input?")

#trainer(model, train_dataset, 2, 2, train_collate, val_dataset=val_dataset, val_steps= 100)

