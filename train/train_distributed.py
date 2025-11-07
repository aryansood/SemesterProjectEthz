from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import torch
from transformers import get_cosine_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import os
from torch.utils.tensorboard import SummaryWriter


#torchrun --nnodes=2 --nproc_per_node=2 --node_rank=0 --master_addr=10.205.9.16 --master_port=29501 post_fine_tune_waymo.py
#tensorboard --logdir=logs/fit --port=6006
#torchrun --nnodes=1 --nproc_per_node=2 --node_rank=0 --master_port=29501 post_fine_tune_waymo.py

def set_up_dist():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return local_rank, rank, world_size

def model_to_dp(model, local_rank):
    model.to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    return model

def dataloader_dist(dataset, batch_size, collate_fn = None):
    train_sampler = DistributedSampler(dataset)
    training_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn= collate_fn
    )
    return train_sampler, training_loader

def get_optimizer(model, optimizer_class=torch.optim.AdamW, lr=1e-5, weight_decay=0.0):
    return optimizer_class(model.parameters(), lr=lr, weight_decay=weight_decay)

def get_scheduler(optimizer, scheduler_fn=get_cosine_schedule_with_warmup, num_warmup_steps=0, num_training_steps=1000):
    return scheduler_fn(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

def validation(model, val_loader, max_step_val = 30):
    loss_sum = 0
    for step_val, batch in enumerate(tqdm(val_loader), start=1):               
        loss_sum+=model.module.validate(batch)
        if(step_val == max_step_val):
            break
    loss_avg = loss_sum/max_step_val
    return loss_avg
    
def trainer(
    model, 
    dataset, 
    batch_size, 
    num_epoch, 
    collate_fn=None,
    optimizer_class=torch.optim.AdamW,
    lr=1e-4,
    weight_decay=0.0,
    scheduler_fn=get_cosine_schedule_with_warmup,
    val_dataset = None,
    val_steps = None,
    max_val_step = 10,
    tensor_board_path = None,
    save_val_dest_path = None,
    save_final_dest_path = None
):
    local_rank, rank, world_size = set_up_dist()
    model = model_to_dp(model, local_rank)
    
    train_sampler, training_loader = dataloader_dist(dataset, batch_size, collate_fn)
    
    optimizer = optimizer_class(model.module.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = scheduler_fn(optimizer, num_warmup_steps=int(0.05*len(training_loader)), num_training_steps=len(training_loader)*num_epoch)

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=10, collate_fn=collate_fn, shuffle=True)
    
    if(rank == 0):
        writer = SummaryWriter(tensor_board_path)

    best_val_loss = float("inf")

    for epoch in range(num_epoch):
        train_sampler.set_epoch(epoch)
        
        for step, batch in enumerate(tqdm(training_loader, disable=(rank != 0)), start=0):
            outputs, loss_value = model(batch)
            loss_value.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            loss_scalar = loss_value.item()

            del outputs, loss_value

            tot_steps = len(training_loader)*epoch + step
            if (rank == 0):
                writer.add_scalar("Loss/train", loss_scalar, tot_steps)
                writer.add_scalar("Val/train", best_val_loss, tot_steps)
            
            if (rank == 0 and val_loader and (tot_steps % val_steps == 0)):
                with torch.no_grad():
                    loss_avg = validation(model, val_loader, max_step_val = max_val_step)
                if(loss_avg < best_val_loss):
                    model.module.save(save_val_dest_path)
                    #torch.save(model.module.dino_traj_decoder.state_dict(), save_val_dest_path)
                    best_val_loss = loss_avg
                torch.cuda.empty_cache()
            print("Train: ", loss_scalar)
            print("Val: ", best_val_loss)

    model.module.save(save_final_dest_path)
    #torch.save(model.module.dino_traj_decoder.state_dict(), save_final_dest_path)