import torch
from model.qwen_model import QwenModel
from dataloader_utils.dataloader_waymo_train import WaymoE2EDatasetTraining, train_collate_fn
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import numpy as np

model = QwenModel(model_to_use=72, device='auto')
# training_dataset = WaymoE2EDatasetTraining('/cluster/scratch/arsood/data_clean', 4)
# training_loader = DataLoader(training_dataset, batch_size=1, shuffle=False, collate_fn=train_collate_fn)
dir_to_save = '/cluster/scratch/arsood/data_strings_train'

index_arr = [6, 8, 10, 12, 14, 16, 19, 20, 21]
batch_size_n = 8
for index_el in index_arr:

    training_dataset = WaymoE2EDatasetTraining('/cluster/scratch/arsood/data_clean', 4, self_cur_idx=index_el)
    training_loader = DataLoader(training_dataset, batch_size=batch_size_n, shuffle=False, collate_fn=train_collate_fn)
    print(index_el)
    for batch in tqdm(training_loader, desc="Loading batches"):
        out = model.generate(batch[5], None, batch[0])
        for i in range(0, len(batch[6])):
            name, ext = os.path.splitext(batch[6][i])
            filename_save = os.path.join(os.path.join(dir_to_save, batch[7][i]), name)
            np.save(filename_save, np.array(out[i]))