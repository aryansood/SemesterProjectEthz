from dataloader_utils.dataloader_waymo import WaymoE2EDatasetTrainingAnnotated, train_collate_waymo
from dataloader_utils.dataloader_covla import CovlaDatasetTrainingAnnotated, train_collate_covla
from dataloader_utils.dataloader_val_waymo import WaymoE2EDatasetVal, val_collate_waymo
from torch.utils.data import DataLoader
from config import data_waymo_train, data_covla, data_waymo_val
from tqdm import tqdm
import numpy as np
from model.qwen_model_virtual_tokens import QwenModelVirtualTokens
from model.qwen_model_base import QwenBaseModel
import torch
from transformers import get_cosine_schedule_with_warmup


model = QwenBaseModel(path_checkpoint='/cluster/scratch/arsood/qwen_base_model_save', is_lora_config = True)

training_data = CovlaDatasetTrainingAnnotated(data_covla.video, data_covla.state, 4, is_fut_traj=True)
training_loader = DataLoader(training_data, batch_size=2, shuffle=True, collate_fn=train_collate_covla)

optimizer = torch.optim.AdamW(model.peft_model.parameters(), lr=2e-4, weight_decay=0.0)

num_training_steps = 25004
num_warmup_steps = int(0.05 * num_training_steps)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

loss_curr = 1000
for batch in tqdm(training_loader):
    outputs, loss_value = model.forward(batch)
    loss_value.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    if(loss_value.value < loss_curr):
        model.save('/cluster/scratch/arsood/qwen_base_model_save')
        loss_curr = loss_value.value