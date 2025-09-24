# import torch
# from model.qwen_model import QwenModel
# from dataloader_utils.dataloader_waymo_train import WaymoE2EDatasetTraining, train_collate_fn
# from tqdm import tqdm
# from torch.utils.data import DataLoader
# import os
# import numpy as np

# #model = QwenModel(model_to_use=72, device='auto')
# dir_to_save = '/cluster/scratch/arsood/data_strings_train'

# index_arr = [6, 8, 10, 12, 14, 16, 19, 20, 21]
# batch_size_n = 8
# for index_el in index_arr:

#     training_dataset = WaymoE2EDatasetTraining('/cluster/scratch/arsood/data_clean', 4, self_cur_idx=index_el)
#     training_loader = DataLoader(training_dataset, batch_size=batch_size_n, shuffle=False, collate_fn=train_collate_fn)
#     print(index_el)
#     for batch in tqdm(training_loader, desc="Loading batches"):
#         #out = model.generate(batch[5], None, batch[0])
#         print(batch[0][0].shape)
#         for i in range(0, len(batch[6])):
#             name, ext = os.path.splitext(batch[6][i])
#             filename_save = os.path.join(os.path.join(dir_to_save, batch[7][i]), name)
#             #np.save(filename_save, np.array(out[i]))


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


load_path_model = '/cluster/scratch/arsood/qwen_waymo_all_finetune_final_5000'
model_first = QwenFineTunedModelText(cache_model = load_path_model)


training_waymo_annotated = WaymoE2EDatasetTraining(data_waymo_train.data, 4)
training_loader = DataLoader(training_waymo_annotated, batch_size = 30, shuffle=False, collate_fn = train_collate)

path_to_save = "/cluster/scratch/arsood/waymo_annotated_3B"
for epoch in range(0, 30):
    print(epoch)
    for batch in tqdm(training_loader):
        out = model_first.generate(batch)
        for i in range(0, len(batch["file_name"])):
            file_where_to_save = os.path.join(path_to_save, batch["file_name"][i])
            np.save(file_where_to_save, np.array(out[i]))