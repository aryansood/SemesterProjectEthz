# import cv2
# import os
# import imageio.v3 as iio
# from tqdm import tqdm
# import json
# import torch
# from model.qwen_model import QwenModel
# import numpy as np
# from dataloader_utils.dataloader_covla_train import CovlaDatasetTraining, train_collate_covla_fn
# from torch.utils.data import DataLoader
# #model = QwenModel(model_to_use=72, device='auto')
# # training_dataset = WaymoE2EDatasetTraining('/cluster/scratch/arsood/data_clean', 4)
# # training_loader = DataLoader(training_dataset, batch_size=1, shuffle=False, collate_fn=train_collate_fn)
# #dir_to_save = '/cluster/scratch/arsood/data_strings_covla'

# import matplotlib.pyplot as plt
# import numpy as np

# dir_path_videos = '/cluster/scratch/arsood/covla/videos'
# dir_path_states  = '/cluster/scratch/arsood/covla/states'

# make_dir = '/cluster/scratch/arsood/covla_string'

# model = QwenModel(model_to_use=72, device='auto')
# dir_to_save = '/cluster/scratch/arsood/covla_string'

# index_arr = [0, 1, 2, 3, 4]
# batch_size_n = 8
# for index_el in index_arr:

#     covla_dataset = CovlaDatasetTraining(dir_path_videos, dir_path_states,list(range(0,5000)), self_cur_idx=index_el)
#     covla_loader = DataLoader(covla_dataset, batch_size=batch_size_n, shuffle=False, collate_fn=train_collate_covla_fn)
#     print(index_el)
#     for batch in tqdm(covla_loader, desc="Loading batches"):
#         l = 10
#         out = model.generate(batch[1], None, batch[0])
#         for i in range(0, len(batch[2])):
#             name = batch[3][i]
#             filename_save = os.path.join(os.path.join(dir_to_save, batch[2][i]), name)
#             np.save(filename_save, np.array(out[i]))

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


training_covla_annotated = CovlaDatasetTraining(data_covla.video, data_covla.state, 4)
training_loader = DataLoader(training_covla_annotated, batch_size = 30, shuffle=False, collate_fn = train_collate)

path_to_save = "/cluster/scratch/arsood/covla_annotated_3B"

for epoch in range(0, 7):
    for batch in tqdm(training_loader):
        out = model_first.generate(batch)
        for i in range(0, len(batch["file_name"])):
            file_where_to_save = os.path.join(path_to_save, batch["file_name"][i])
            np.save(file_where_to_save, np.array(out[i]))
            



