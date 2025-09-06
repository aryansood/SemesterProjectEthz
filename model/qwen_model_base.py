import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import cv2
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.optim import Adam
from scipy.interpolate import CubicSpline
import re
from peft import PeftModel
import json

class QwenBaseModel(nn.Module):
    def __init__(self, cache_dir = '/cluster/scratch/arsood/cache_hugging_face', local_files_only= True, is_training = True, is_lora_config = True, path_checkpoint = "",device = 'cuda'):
        super(QwenBaseModel, self).__init__()

        model_qwen_select =  "/cluster/scratch/arsood/cache_hugging_face/Qwen2.5-VL-3B-Instruct/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3"
        
        self.device = device

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_qwen_select, torch_dtype=torch.bfloat16, device_map=device)
            #cache_dir = cache_dir, local_files_only= local_files_only)

        self.processor = AutoProcessor.from_pretrained(
            model_qwen_select,
            #cache_dir=cache_dir,
            #local_files_only=local_files_only
            )

        peft_config = LoraConfig(
            lora_alpha=128,
            lora_dropout=0.05,
            r=64,
            bias="none",
            target_modules=["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",], 
            task_type="CAUSAL_LM",
        )

        self.peft_model = self.model
        for param in self.model.parameters():
            param.requires_grad = True
        # if is_lora_config == True:
        #     self.peft_model = get_peft_model(self.model, peft_config)
        # else:
        #     self.peft_model = self.load(path_checkpoint)

        for name, param in self.peft_model.named_parameters():
            if param.requires_grad:
                print(f"{name}: requires_grad = {param.requires_grad}")

       
        # if torch.cuda.device_count() > 1:
        #     self.peft_model = nn.DataParallel(self.peft_model)
        self.peft_model.to(device)
        self.is_training = is_training

    def prepare_input_for_training(self, messages, images, videos):
        
        len_batch = len(messages) 
        self.processor.tokenizer.padding_side = "left"
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt= (not self.is_training)) for msg in messages
        ]
        batch = self.processor(
            text=texts,
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
        )
        if self.is_training:
            labels = batch["input_ids"].clone()
            assistant_start_ids = self.processor.tokenizer.convert_tokens_to_ids('<|im_start|>')
            indices = torch.where(labels == assistant_start_ids)
            len_token_spec = int((indices[0].shape[0])/len_batch)
            for i in range(0, len_batch):
                labels[i][:int(indices[1][len_token_spec*(i+1)-1])] = -100
            labels = labels.to(self.device)
            batch = batch.to(self.device)
            return batch, labels
        
        batch = batch.to(self.device)

        return batch
        
    def loss(self, logits, labels):
        logits = logits.permute(0, 2, 1)
        logits = logits[..., :-1]
        labels = labels[..., 1:]
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        loss_cross = criterion(logits, labels)
        return loss_cross

    def forward(self, x):
        batch, labels = self.prepare_input_for_training(x[0], None, x[1])
        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            outputs = self.peft_model(**batch, output_hidden_states=True)
        if self.is_training:
            loss_value = self.loss(outputs.logits, labels)
            return outputs, loss_value
        
        return outputs
    
    def loss_validation(self, pred, target):
        loss_diff = torch.sqrt(torch.square(pred-target).sum(dim = -1)).mean(dim = 0)
        return loss_diff

    def validate(self, x):
        self.is_training = False
        input_ids =  self.prepare_input_for_training(x[0], None, x[1])
        index_max = np.argmax(np.array(x[3]), axis = 1)
        traj_fut_opt = np.array(x[5])
        l = 0
        with torch.no_grad():
            outputs = self.peft_model.generate(**input_ids, max_new_tokens=400)
            generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids.input_ids, outputs)
            ]
            output_text = self.processor.batch_decode( 
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            traj_list_gt = []
            traj_list_pred = []
            for el in output_text:
                clean_string = str(el).strip()
                if clean_string.startswith("```json"):
                    clean_string = clean_string[len("```json"):].strip()
                if clean_string.endswith("```"):
                    clean_string = clean_string[:-3].strip()
                gt_label = json.loads(str(clean_string))
                arr = np.array(gt_label['traj_fut'])
                index = [0, 3, 7, 11, 15, 19]
                cs_x = CubicSpline(index, arr[...,0])
                cs_y = CubicSpline(index, arr[...,1])
                t_high = np.arange(0, 20)
                x_high = cs_x(t_high)[:, None]
                y_high = cs_y(t_high)[:, None]
                traj_pred = np.concatenate([x_high, y_high], axis = -1)
                traj_rat_best = x[4][l][int(index_max[l])]
                if(traj_rat_best.shape[0] == 21):
                    traj_list_gt.append(traj_fut_opt[l][...,0:2])#traj_rat_best[:-1])
                    traj_list_pred.append(traj_pred)
                l += 1
            traj_list_gt = torch.from_numpy(np.array(traj_list_gt))
            traj_list_pred = torch.from_numpy(np.array(traj_list_pred))
            loss_avg = self.loss_validation(traj_list_pred, traj_list_gt)
            
        self.is_training = True
        return loss_avg

    def save(self, path_to_save):
        self.peft_model.save_pretrained(path_to_save, safe_serialization=True)

    def load(self, path_checkpoint):
        model_combined = PeftModel.from_pretrained(self.model, path_checkpoint)
        model_combined.train()
        model_combined.enable_adapter_layers()
        return model_combined
    
    def generate(self, messages, images, videos):
        pass
    
    