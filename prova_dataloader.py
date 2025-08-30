from dataloader_utils.dataloader_waymo import WaymoE2EDatasetTrainingAnnotated, train_collate_fn
from dataloader_utils.dataloader_covla import CovlaDatasetTrainingAnnotated
from dataloader_utils.dataloader_val_waymo import WaymoE2EDatasetVal
from torch.utils.data import DataLoader
from config import data_waymo_train, data_covla, data_waymo_val
from tqdm import tqdm
import numpy as np
from model.qwen_model_virtual_tokens import QwenModelVirtualTokens


model = QwenModelVirtualTokens()

training_data = WaymoE2EDatasetTrainingAnnotated(data_waymo_train.waymo_data, 4)
training_loader = DataLoader(training_data, batch_size=4, shuffle=False, collate_fn=train_collate_fn)

for batch in tqdm(training_loader):
    model.forward(batch)
    inupt_d = input("ciao")