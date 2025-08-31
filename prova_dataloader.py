from dataloader_utils.dataloader_waymo import WaymoE2EDatasetTrainingAnnotated, train_collate_waymo
from dataloader_utils.dataloader_covla import CovlaDatasetTrainingAnnotated, train_collate_covla
from dataloader_utils.dataloader_val_waymo import WaymoE2EDatasetVal
from torch.utils.data import DataLoader
from config import data_waymo_train, data_covla, data_waymo_val
from tqdm import tqdm
import numpy as np
from model.qwen_model_virtual_tokens import QwenModelVirtualTokens


model = QwenModelVirtualTokens()

training_data = CovlaDatasetTrainingAnnotated(data_covla.covla_video, data_covla.covla_state, 4)
training_loader = DataLoader(training_data, batch_size=2, shuffle=False, collate_fn=train_collate_covla)

for batch in tqdm(training_loader):
    model.forward(batch)
    inupt_d = input("ciao")