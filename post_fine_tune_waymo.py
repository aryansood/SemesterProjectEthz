from dataloader_utils.dataloader_waymo import WaymoE2EDatasetTrainingAnnotated, WaymoE2EDatasetTraining, train_collate_waymo
from dataloader_utils.dataloader_val_waymo import WaymoE2EDatasetVal, val_collate_waymo
from torch.utils.data import DataLoader, ConcatDataset
from config import data_waymo_train, data_covla, data_waymo_val
from tqdm import tqdm
from model.qwen_model_base import QwenBaseModel
import torch
from transformers import get_cosine_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
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

training_waymo_annotated = WaymoE2EDatasetTrainingAnnotated(data_waymo_train.data, 4, is_fut_traj=True)
# training_waymo_1 = WaymoE2EDatasetTraining(data_waymo_train.data, 4)
# training_waymo_2 = WaymoE2EDatasetTraining(data_waymo_train.data, 4)
# training_waymo_3 = WaymoE2EDatasetTraining(data_waymo_train.data, 4)
# training_data = ConcatDataset([training_waymo_annotated, training_waymo_1, training_waymo_2, training_waymo_3])

###########################################################################
model = QwenBaseModel(path_checkpoint='/cluster/scratch/arsood/qwen_covla_final_2', is_lora_config = False).to(local_rank)
model = DDP(model, device_ids=[local_rank], output_device=local_rank)
train_sampler = DistributedSampler(training_waymo_annotated)
training_loader = DataLoader(
    training_waymo_annotated,
    batch_size=2,
    sampler=train_sampler,
    collate_fn=train_collate_waymo
)
if rank == 0:
    val_dataset = WaymoE2EDatasetVal(data_waymo_val.data, 4)
    val_loader = DataLoader(val_dataset, batch_size=70, shuffle=False, collate_fn=val_collate_waymo)
    writer = SummaryWriter("/cluster/scratch/arsood/run_tensorboard_waymo_5")
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
loss_norm = 1000
step = 0
for epoch in range(0,2):
    train_sampler.set_epoch(epoch)
    for batch in tqdm(training_loader, disable=(rank != 0)):
        outputs, loss_value = model.module.forward(batch)
        loss_value.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        if rank == 0:
            print(loss_value.item())
            if(loss_value.item() < loss_norm):
                #model.module.save('/cluster/scratch/arsood/qwen_waymo_all_finetune_3')
                loss_norm = loss_value.item()
            # if(step % 1500 == 0):
            #     num_val_steps = 0
            #     loss_avg = torch.zeros(20)
            #     for batch_val in val_loader:
            #         loss_avg += model.module.validate(batch_val)
            #         num_val_steps+=1
            #     loss_avg = loss_avg/num_val_steps
            #     if(loss_avg[11].item() < loss_curr):
            #         model.module.save('/cluster/scratch/arsood/qwen_waymo')
            #         loss_curr = loss_avg[11].item()
            writer.add_scalar("Loss/train", loss_value.item(), step)
            #writer.add_scalar("Val/3Ade", loss_curr, step)
            step+=1
model.module.save('/cluster/scratch/arsood/qwen_waymo_all_finetune_final_3')
dist.destroy_process_group()
