from dataloader_utils.dataloader_waymo import WaymoE2EDatasetTrainingAnnotated, train_collate_waymo
from dataloader_utils.dataloader_covla import CovlaDatasetTrainingAnnotated, CovlaDatasetTraining, train_collate_covla
from dataloader_utils.dataloader_val_waymo import WaymoE2EDatasetVal, val_collate_waymo
from torch.utils.data import DataLoader, ConcatDataset
from config import data_waymo_train, data_covla, data_waymo_val
from tqdm import tqdm
import numpy as np
from model.qwen_model_virtual_tokens import QwenModelVirtualTokens
from model.qwen_model_base import QwenBaseModel
import torch
from transformers import get_cosine_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from huggingface_hub import snapshot_download
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import os

##################################################################
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl", init_method="env://")
rank = dist.get_rank()
world_size = dist.get_world_size()
##################################################################


training_covla_annotated = CovlaDatasetTrainingAnnotated(data_covla.video, data_covla.state, 4, is_fut_traj=True)
training_covla_1 = CovlaDatasetTraining(data_covla.video, data_covla.state)
training_covla_2 = CovlaDatasetTraining(data_covla.video, data_covla.state)
training_data = ConcatDataset([training_covla_annotated, training_covla_1, training_covla_2])
#training_loader = DataLoader(training_data, batch_size=3, shuffle=True, collate_fn=train_collate_covla)


###########################################################################
model = QwenBaseModel(path_checkpoint='', is_lora_config = True).to(local_rank)
model = DDP(model, device_ids=[local_rank], output_device=local_rank)
train_sampler = DistributedSampler(training_data)
training_loader = DataLoader(
    training_data,
    batch_size=4,
    sampler=train_sampler,
    collate_fn=train_collate_covla
)
if rank == 0:
    # val_dataset = WaymoE2EDatasetVal(data_waymo_val.data, 4)
    # val_loader = DataLoader(val_dataset, batch_size=70, shuffle=False, collate_fn=train_collate_covla)
    writer = SummaryWriter("/cluster/scratch/arsood/run_tensorboard_covla_3")
else:
    val_loader = None
    writer = None
#############################################################################

optimizer = torch.optim.AdamW(model.module.peft_model.parameters(), lr=1e-5, weight_decay=0.0)
num_warmup_steps = int(0.05 * len(training_loader.dataset))
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=len(training_loader.dataset)
)

loss_curr = 1000
step = 0

for epoch in range(0,1):
    train_sampler.set_epoch(epoch)
    for batch in tqdm(training_loader, disable=(rank != 0)):
        outputs, loss_value = model.forward(batch)
        loss_value.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        if rank == 0:
            writer.add_scalar("Loss/train", loss_value.item(), step)
            if(loss_value.item() < loss_curr):
                model.module.save('/cluster/scratch/arsood/qwen_covla_2')
                loss_curr = loss_value.item()
            step += 1
model.module.save('/cluster/scratch/arsood/qwen_covla_final_2')

# model_dir = snapshot_download(
#     repo_id='Qwen/Qwen2.5-VL-3B-Instruct',
#     cache_dir='/cluster/scratch/arsood/cache_hugging_face/Qwen2.5-VL-3B-Instruct'
# )

##torchrun --nnodes=2 --nproc_per_node=2 --node_rank=0 --master_addr=10.205.9.16 --master_port=29501 post_fine_tune_waymo.py