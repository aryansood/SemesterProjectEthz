from dataloader_utils.dataloader_waymo import WaymoE2EDatasetTrainingAnnotated, train_collate_waymo
from dataloader_utils.dataloader_covla import CovlaDatasetTrainingAnnotated, train_collate_covla
from dataloader_utils.dataloader_val_waymo import WaymoE2EDatasetVal
from torch.utils.data import DataLoader
from config import data_waymo_train, data_covla, data_waymo_val
from tqdm import tqdm
import numpy as np
from model.qwen_model_virtual_tokens import QwenModelVirtualTokens
import torch


model = QwenModelVirtualTokens()

training_data = CovlaDatasetTrainingAnnotated(data_covla.covla_video, data_covla.covla_state, 4)
training_loader = DataLoader(training_data, batch_size=2, shuffle=True, collate_fn=train_collate_covla)

optimizer = torch.optim.Adam(model.peft_model.parameters(), lr=1e-4)

for batch in tqdm(training_loader):
    outputs, loss_value = model.forward(batch)
    loss_value.backward()
    emb = model.peft_model.get_input_embeddings().weight
    emb.grad[0:-20] = 0
    optimizer.step()
    optimizer.zero_grad()
    print(loss_value)