from model.model_dino_qwen_virtual import DinoEncoder, PipelineDino
import torch
from transformers import CLIPModel, CLIPProcessor
from dataloader_utils.dataloader_waymo import WaymoE2EDatasetTrainingAnnotated, WaymoE2EDatasetTraining
from dataloader_utils.dataloader_covla import CovlaDatasetTrainingAnnotated, CovlaDatasetTraining
from torch.utils.data import DataLoader, ConcatDataset
from dataloader_utils.collate_functions import train_collate
from config import data_waymo_train, data_covla, data_waymo_val
from torch.utils.data import DataLoader
from dataloader_utils.dataloader_val_waymo import WaymoE2EDatasetVal, val_collate_waymo
from dataloader_utils.collate_functions import train_collate
from tqdm import tqdm
from torch.utils.data import random_split
from train.train_distributed import trainer
import matplotlib.pyplot as plt
import os

# save_val_dest_path = '/cluster/scratch/arsood/qwen_with_fut_traj_val_3'
# text_gen_encoder = QwenEncoder(cache_model = save_val_dest_path)

#
# val_loader = DataLoader(val_dataset, batch_size=1, shuffle = False, collate_fn = train_collate)
model = PipelineDino()

# training_covla_annotated = CovlaDatasetTrainingAnnotated(data_covla.video, data_covla.state, 1, is_fut_traj=True)
# training_waymo_annotated = WaymoE2EDatasetTrainingAnnotated(data_waymo_train.data, 1, is_fut_traj=True)

training_covla = CovlaDatasetTraining(data_covla.video, data_covla.state, 1)
training_waymo = WaymoE2EDatasetTraining(data_waymo_train.data, 1)
training_data = ConcatDataset([training_covla, training_waymo])

#train_loader = DataLoader(training_data, batch_size=20, shuffle = True, collate_fn = train_collate)

# dataset_size = len(training_data)
# val_size = int(0.1 * dataset_size)
# train_size = dataset_size - val_size

# torch.manual_seed(42)
# train_dataset, val_dataset = random_split(training_data, [train_size, val_size])
val_dataset = WaymoE2EDatasetVal(data_waymo_val.data, 1)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle = False, collate_fn = train_collate)

save_val_dest_path = '/cluster/scratch/arsood/dino_qwen_val/model.pth'
save_final_dest_path = '/cluster/scratch/arsood/dino_qwen_final_3/model.pth'
tensor_board_path = '/cluster/scratch/arsood/dino_qwen_6'
# os.makedirs('/cluster/scratch/arsood/dino_qwen_val_3/', exist_ok=True)
# os.makedirs('/cluster/scratch/arsood/dino_qwen_final_3/', exist_ok=True)

model.dino_traj_decoder.load_state_dict(torch.load(save_val_dest_path, map_location=torch.device("cuda")))
model.to('cuda')

# trainer(model, training_data, 30, 50, train_collate, val_dataset=None, val_steps= 50, tensor_board_path = tensor_board_path, save_val_dest_path=save_val_dest_path, save_final_dest_path = save_final_dest_path)

for batch in tqdm(val_loader):
    # indices = [0, 3, 7, 11, 15, 19]
    # with torch.no_grad():
    out = model(batch)
    print(out[1])
    plt.figure()
    traj_pred = out[0].detach().cpu().numpy()
    plt.xlim(-20,20)
    plt.plot(traj_pred[0][...,1], traj_pred[0][...,0], color="blue")
    plt.plot(batch["next_state_traj"][0][...,1], batch["next_state_traj"][0][...,0], color="red")
    plt.savefig("/cluster/home/arsood/Semester_Project_Official/colo.png")
    input_l = input("lol")

# for batch in tqdm(train_loader):
#     print(batch["front_images"][0][0].shape)
#     out = text_gen_encoder.forward(batch)
#     inpu_l = input("COntue")
