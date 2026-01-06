from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset
from torch import nn 
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import torch
from dataloader_utils_2.waymo_dataloader_split import WaymoE2EDatasetTraining
import random
from torch.utils.data import DataLoader
from config import data_waymo_train, data_covla, data_waymo_val
from scipy.interpolate import CubicSpline
import re
import json
import numpy as np
from transformers import TrainerCallback

def is_valid_traj_fut(s):
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        return False
    if not isinstance(data, dict):
        return False

    if "traj_fut" not in data:
        return False

    traj = data["traj_fut"]
    if not isinstance(traj, list):
        return False
    
    if len(traj) != 19:
        return False
    
    for point in traj:
        if (
            not isinstance(point, list)
            or len(point) != 2
            or not all(isinstance(v, (int, float)) for v in point)
        ):
            return False

    return True

def json_to_traj(el):
    converted_bool = False
    traj_pred = None
    clean_string = str(el).strip()

    if clean_string.startswith("```json"):
        clean_string = clean_string[len("```json"):].strip()
    if clean_string.endswith("```"):
        clean_string = clean_string[:-3].strip()
    gt_label = json.loads(str(clean_string))
    # traj_pred = np.array(gt_label['traj_fut'])
    if 'traj_fut' in gt_label:
        arr = np.array(gt_label['traj_fut'])
        # index = [0, 3, 7, 11, 15, 19]
        index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19]
        cs_x = CubicSpline(index, arr[...,0])
        cs_y = CubicSpline(index, arr[...,1])
        t_high = np.arange(0, 20)
        x_high = cs_x(t_high)[:, None]
        y_high = cs_y(t_high)[:, None]
        traj_pred = np.concatenate([x_high, y_high], axis = -1)
    return traj_pred

def ade_at_5(predict, target):
    score = np.sqrt(np.square((predict-target).sum(axis = -1))).mean()
    return score 

def reward_grpo(completions, next_state_traj,**kwargs):
    score_list = []
    for index_num, text_compl in enumerate(completions):
        val = is_valid_traj_fut(text_compl[0]['content'])
        if(val):
            traj_predict = json_to_traj(text_compl[0]['content'])
            score = ade_at_5(traj_predict, next_state_traj[index_num][..., 0:2])
            score_list.append(-score)
        else:
            score_list.append(-100)
    print(score_list)
    print(completions)
    return score_list

# save_final_dest_path = '/cluster/scratch/arsood/qwen_fine_tune_covla_wcot_final'
processor_config = "/cluster/scratch/arsood/Qwen_2_5_vlm"
save_final_dest_path = '/cluster/scratch/arsood/qwen_fine_tune_covla_wcot_final_2'

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            save_final_dest_path, device_map='cuda')

processor = AutoProcessor.from_pretrained(
            processor_config
            )

num_generations = 8
per_device_train_batch_size = 8
data_set_grpo = WaymoE2EDatasetTraining(data_waymo_train.data, 1, processor, is_Grpo = True, num_generations = num_generations)

output_dir = "/cluster/scratch/arsood/models/Qwen2.5-3B-GRPO_Full_Traj"

for param in model.parameters():
    param.requires_grad = True

training_args = GRPOConfig(
    learning_rate=1e-6,
    num_train_epochs=2,
    per_device_train_batch_size=per_device_train_batch_size,
    max_completion_length=1024,
    num_generations=num_generations,
    max_prompt_length=2048,
    fp16=True,
    output_dir=output_dir,                        
    logging_steps=1,
    temperature = 0.7,
    top_p=0.9,
    save_strategy="steps",
    save_steps=250,
    save_total_limit=3,
    report_to="wandb",
    run_name="grpo-exp-Full_Traj",
    repetition_penalty = 1,
    # gradient_accumulation_steps = 4,
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_grpo,
    args=training_args,
    train_dataset=data_set_grpo,
    processing_class = processor,
)
trainer.train()



# def compare_traj_fun(predict_traj , gt_traj_list, score_list):
#     # for index_score, el_traj in enumerate(gt_traj_list):
#     #     min_dim = min(predict_traj.shape[0], el_traj.shape[0])
#     #     reduced_min_traj = predict_traj[:min_dim]
#     #     reduced_rater_traj = el_traj[:min_dim]
#     #     diff_traj = reduced_min_traj- reduced_rater_traj
#     #     ade_5_sec = np.sqrt(np.square(diff_traj).sum(axis = -1)).mean()
#     #     print(el_traj.shape, ade_5_sec)
