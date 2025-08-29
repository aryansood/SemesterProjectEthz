from dataloader_utils.dataloader_waymo import WaymoE2EDatasetTrainingAnnotated
from dataloader_utils.dataloader_covla import CovlaDatasetTrainingAnnotated
from dataloader_utils.dataloader_val_waymo import WaymoE2EDatasetVal
from torch.utils.data import DataLoader
from config import data_waymo_train, data_covla, data_waymo_val
from tqdm import tqdm
import numpy as np
import zipfile
import json


# training_dataset = CovlaDatasetTrainingAnnotated(data_covla.covla_video, data_covla.covla_state, 4)
# training_loader = DataLoader(training_dataset, 1, shuffle=False)
val_dataset = WaymoE2EDatasetVal(data_waymo_val.waymo_data, 4)
val_loader = DataLoader(val_dataset, 1, shuffle=False)
for batch in tqdm(val_loader):
    l = 10
    input_l = input("Contiue?")