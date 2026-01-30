from dataloader_utils_2.waymo_dataloader_split import WaymoE2EDatasetTraining
from dataloader_utils_2.covla_dataloader_split import CovlaDatasetTraining
from dataloader_utils_2.collate_fn_train import train_collate
from dataloader_utils_2.waymo_val_dataloader import WaymoE2EDatasetVal, collate_val
from torch.utils.data import DataLoader, ConcatDataset
from config import data_waymo_train, data_covla, data_waymo_val
from tqdm import tqdm
import numpy as np
import torch 
from model.qwen_finetuned_text import QwenFineTunedModelText
from model.qwen_cot_latent import QwenLatentCot
from utils.utils_camera_print import validate_proj_save


dataset_waymo_val = WaymoE2EDatasetVal(data_waymo_val.data, 4)

dataloader_val = DataLoader(dataset_waymo_val, batch_size = 20, shuffle=False, collate_fn=collate_val)



prcoessor_config = "/cluster/scratch/arsood/Qwen_2_5_vlm"
model_to_load = "/cluster/scratch/arsood/models/Qwen2.5-3B-GRPO_Full_Traj_3/checkpoint-4074"
model = QwenFineTunedModelText(model_to_load, prcoessor_config)

# prcoessor_config = "/cluster/scratch/arsood/Qwen_2_5_vlm"
# model_to_load_2 = "/cluster/scratch/arsood/Qwen_Fine_Tune_Waymo_Data_Final"
# path_val_virt = "/cluster/scratch/arsood/cot_latent_val_spline/cot_latent_val_virt.pt"
# path_val_linear = "/cluster/scratch/arsood/cot_latent_val_spline/cot_latent_val_linear.pt"
# model = QwenLatentCot(model_to_load_2, prcoessor_config)
# model.load_model(path_val_virt, path_val_linear)


path_to_save_results = "/cluster/scratch/arsood/full_traj_grpo_new/results.npy"
# path_to_save_results = "/cluster/scratch/arsood/5_sec_grpo_new/results.npy"
# # path_to_save_results = "/cluster/scratch/arsood/vlm_cot_5/results.npy"
# path_to_save_results = "/cluster/scratch/arsood/vlm_latent_cot_5/results.npy"


validate_proj_save(dataloader_val, model, path_to_save_results)



# ade_3 = 0
# ade_5 = 0
# count = 0
# for batch in tqdm(dataloader_val):
#     with torch.no_grad():
#         result_val, lol = model.generate_traj_val(batch)
#         ade_3 += torch.mean(result_val[...,0:12]).item()
#         ade_5 += torch.mean(result_val).item()
#         count+=1
#         print(ade_3/count)
#         print(ade_5/count)
# print(ade_3/count)
# print(ade_5/count)