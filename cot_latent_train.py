from dataloader_utils_2.waymo_dataloader_split import WaymoE2EDatasetTraining
from dataloader_utils_2.covla_dataloader_split import CovlaDatasetTraining
from dataloader_utils_2.collate_fn_train import train_collate
from dataloader_utils_2.waymo_val_dataloader import WaymoE2EDatasetVal, collate_val
from torch.utils.data import DataLoader, ConcatDataset
from config import data_waymo_train, data_covla, data_waymo_val
from tqdm import tqdm
import numpy as np
import torch
import os
from train.train_distributed import trainer
from torch.utils.data import random_split
from config import data_waymo_train
from model.qwen_cot_latent import QwenLatentCot
from torch.optim import AdamW
import wandb
from utils.utils_camera_print import validate_proj_save

run = wandb.init(
    project="cot_latent_check",
)

dataset_waymo = WaymoE2EDatasetTraining(data_waymo_train.data, 4)
dataset_waymo_val = WaymoE2EDatasetVal(data_waymo_val.data, 4)

dataset = ConcatDataset([dataset_waymo])
dataloader_train = DataLoader(dataset, batch_size = 3, shuffle=True, collate_fn=train_collate)
dataloader_val = DataLoader(dataset_waymo_val, batch_size = 3, shuffle=True, collate_fn=collate_val)

# path_val_virt = "/cluster/scratch/arsood/cot_latent_val_spline/cot_latent_val_virt.pt"
# path_val_linear = "/cluster/scratch/arsood/cot_latent_val_spline/cot_latent_val_linear.pt"

# path_final_virt =  "/cluster/scratch/arsood/cot_latent_final_spline/cot_latent_final_virt.pt"
# path_final_linear =  "/cluster/scratch/arsood/cot_latent_final_spline/cot_latent_final_linear.pt"

path_val_virt = "/cluster/scratch/arsood/cot_latent_val_spline_9/cot_latent_val_virt.pt"
path_val_linear = "/cluster/scratch/arsood/cot_latent_val_spline_9/cot_latent_val_linear.pt"

path_final_virt =  "/cluster/scratch/arsood/cot_latent_final_spline_9/cot_latent_final_virt.pt"
path_final_linear =  "/cluster/scratch/arsood/cot_latent_final_spline_9/cot_latent_final_linear.pt"

# path_val = "/cluster/scratch/arsood/cot_latent_val_spline_8/whole_model.pt"
# path_final = "/cluster/scratch/arsood/cot_latent_val_spline_8_final/whole_model.pt"

num_step_val = 100
step = 0
os.makedirs(os.path.dirname(path_val_virt), exist_ok=True)
os.makedirs(os.path.dirname(path_final_virt), exist_ok=True)

prcoessor_config = "/cluster/scratch/arsood/Qwen_2_5_vlm"
model_to_load_2 = "/cluster/scratch/arsood/Qwen_Fine_Tune_Waymo_Data_Final"
model = QwenLatentCot(model_to_load_2, prcoessor_config)
# model.load(path_val_virt, path_val_linear)
optmizer = AdamW(model.parameters(), lr=2e-4)
# model.load(path_val_virt, path_val_linear)

# path_to_save_results = "/cluster/scratch/arsood/cot_latent_val_results_spline/results.npy"
# validate_proj_save(dataloader_val, model, path_to_save_results)

# ade_3 = 0
# ade_5 = 0
# count = 0
# for batch in tqdm(dataloader_val):
#     with torch.no_grad():
#         result_val = model.validate(batch)
#         ade_3 += torch.mean(result_val[...,0:12]).item()
#         ade_5 += torch.mean(result_val).item()
#         count+=1
#         print(ade_3/count)
#         print(ade_5/count)
# print(ade_3/count)
# print(ade_5/count)

cur_val = 100

for epoch in range(0, 250):
    for batch in tqdm(dataloader_train):
        optmizer.zero_grad()
        output, loss = model(batch)
        loss.backward()
        print("Loss:", loss.detach().item())
        run.log({"loss": loss.detach().item()})
        del output, loss
        before_el = model.model.get_input_embeddings().weight.clone().detach()
        optmizer.step()
        torch.cuda.empty_cache()
        with torch.no_grad():
            model.model.get_input_embeddings().weight[:-model.cot_latent_num_tokens] = before_el[:-model.cot_latent_num_tokens]
        if step % num_step_val == 0:
            with torch.no_grad():
                ade_3 = 0
                ade_5 = 0
                count = 0
                for batch_val in tqdm(dataloader_val):
                    result_val = model.validate(batch_val)
                    ade_3 += torch.mean(result_val[...,0:12]).item()
                    ade_5 += torch.mean(result_val).item()
                    count+=1
                run.log({"ade@3": ade_3/count})
                run.log({"ade@5": ade_5/count})
                if(cur_val > (ade_5/count)):
                    cur_val = ade_5/count
                    model.save(path_val_virt, path_val_linear)
                    #model.save_all(path_val)
            torch.cuda.empty_cache()
        step = step+1
model.save(path_final_virt, path_final_linear)
# model.save_all(path_final)

"""
for name, p in model.named_parameters():
    if p.grad is not None and p.grad.norm() > 0:
        print("GRAD:", name)
"""