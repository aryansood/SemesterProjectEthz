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

class QwenBaseModel(nn.Module):
    def __init__(self, cache_dir = '/cluster/scratch/arsood/cache_hugging_face', local_files_only= True, is_training = True, is_lora_config = True, path_checkpoint = "",device = 'cuda'):
        super(QwenBaseModel, self).__init__()

        model_qwen_select =  "Qwen/Qwen2.5-VL-3B-Instruct"
        
        self.device = device

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_qwen_select, torch_dtype=torch.bfloat16, device_map=device,
            cache_dir = cache_dir, local_files_only= local_files_only)

        self.processor = AutoProcessor.from_pretrained(
            model_qwen_select,
            cache_dir=cache_dir,
            local_files_only=local_files_only
            )

        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.05,
            r=16,
            bias="none",
            target_modules=["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",], 
            task_type="CAUSAL_LM",
        )

        self.peft_model = None
        if is_lora_config == True:
            self.peft_model = get_peft_model(self.model, peft_config)
        else:
            self.peft_model = self.load(path_checkpoint)

        # for name, param in self.peft_model.named_parameters():
        #     if param.requires_grad:
        #         print(f"{name}: requires_grad = {param.requires_grad}")

       
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

    def validate(self, x):
        self.is_training = False
        input_ids =  self.prepare_input_for_training(x[0], None, x[1])
        with torch.no_grad():
            outputs = self.peft_model.generate(**input_ids, max_new_tokens=400)
            generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids.input_ids, outputs)
            ]
            output_text = self.processor.batch_decode( 
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            print(output_text)
            # for el_out in outputs:
            #     s_clean = el_out.strip('{}')
            #     numbers = list(map(float, re.findall(r"[-+]?\d*\.\d+|\d+", s_clean)))
            #     arr = np.array(numbers).reshape(-1, 2)
            #     index = [0, 3, 7, 11, 15, 19]
            #     cs_x = CubicSpline(index, arr[...,0])
            #     cs_y = CubicSpline(index, arr[...,1])
            #     t_high = np.arange(0, 20)
            #     x_high = cs_x(t_high)[:, None]
            #     y_high = cs_y(t_high)[:, None]
            #     #x = next_state_traj[...,0]
            #     #y = next_state_traj[...,1]
        self.is_training = True

    def save(self, path_to_save):
        self.peft_model.save_pretrained(path_to_save)

    def load(self, path_checkpoint):
        model_combined = PeftModel.from_pretrained(self.model, path_checkpoint)
        return model_combined
    
    def generate(self, messages, images, videos):
        pass
    
    