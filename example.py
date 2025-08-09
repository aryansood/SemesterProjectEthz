import torch
from model.qwen_model import QwenModel
from dataloader_utils.dataloader_waymo_train import WaymoE2EDatasetTraining, train_collate_fn
from tqdm import tqdm
from torch.utils.data import DataLoader

model = QwenModel()

training_dataset = WaymoE2EDatasetTraining('/cluster/scratch/arsood/data_clean_val', 5)
training_loader = DataLoader(training_dataset, batch_size=1, shuffle=True, collate_fn=train_collate_fn)

for batch in tqdm(training_loader, desc="Loading batches"):
    out = model.generate(batch[5], None, batch[0])
    print(out)
    a = input("Contnue?")
