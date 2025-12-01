from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset
from torch import nn 
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import torch
from dataloader_utils.dataloader_val_waymo import WaymoE2EDatasetVal
import random
from torch.utils.data import DataLoader
from config import data_waymo_train, data_covla, data_waymo_val
from scipy.interpolate import CubicSpline
import re
import json
import numpy as np

def json_to_traj(el):
    converted_bool = False
    traj_pred = None
    clean_string = str(el).strip()

    if clean_string.startswith("```json"):
        clean_string = clean_string[len("```json"):].strip()
    if clean_string.endswith("```"):
        clean_string = clean_string[:-3].strip()
    gt_label = json.loads(str(clean_string))

    if 'traj_fut' in gt_label:
        arr = np.array(gt_label['traj_fut'])
        if(arr.shape[0] == 6):
            converted_bool = True
            index = [0, 3, 7, 11, 15, 19]
            cs_x = CubicSpline(index, arr[...,0])
            cs_y = CubicSpline(index, arr[...,1])
            t_high = np.arange(0, 20)
            x_high = cs_x(t_high)[:, None]
            y_high = cs_y(t_high)[:, None]
            traj_pred = np.concatenate([x_high, y_high], axis = -1)
    return traj_pred, converted_bool

def compare_traj_fun(predict_traj , gt_traj_list, score_list):
    for index_score, el_traj in enumerate(gt_traj_list):
        min_dim = min(predict_traj.shape[0], el_traj.shape[0])
        reduced_min_traj = predict_traj[:min_dim]
        reduced_rater_traj = el_traj[:min_dim]
        diff_traj = reduced_min_traj- reduced_rater_traj
        ade_5_sec = np.sqrt(np.square(diff_traj).sum(axis = -1)).mean()
        print(el_traj.shape, ade_5_sec)

def reward_grpo(completions, traj_rater, traj_rater_score, **kwargs):
    for index_num, text_compl in enumerate(completions):
        traj_predict, converted_bool = json_to_traj(text_compl)
        if(converted_bool):
            compare_traj_fun(traj_predict, traj_rater[index_num], traj_rater_score[index_num])
            input_l = input("ciao?")
        else:
            rewards_to_give.append(-10)
    rewards_to_give = [random.uniform(-20, 20) for el in completions]
    return rewards_to_give

save_val_dest_path = '/cluster/scratch/arsood/qwen_with_fut_traj_wo_past_traj_6'
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            save_val_dest_path, device_map='cuda') #dtype=torch.float16 torch.bfloat16

processor = AutoProcessor.from_pretrained(
            "/cluster/scratch/arsood/models/Qwen2.5-VL-3B-Instruct"
            )
data_set_grpo = WaymoE2EDatasetVal(data_waymo_val.data, 1, processor)

output_dir = "/cluster/scratch/arsood/models/Qwen2.5-3B-GRPO"

for param in model.parameters():
    param.requires_grad = True

training_args = GRPOConfig(
    learning_rate=2e-5,
    num_train_epochs=1,
    per_device_train_batch_size=20,
    max_completion_length=1024,
    num_generations=4,
    max_prompt_length=2048,
    fp16=True,
    output_dir=output_dir,                        
    logging_steps=1,
    temperature = 0.7,
    top_p=0.9,
    # Hub integration
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_grpo,
    args=training_args,
    train_dataset=data_set_grpo,
    processing_class = processor,
)
trainer.train()