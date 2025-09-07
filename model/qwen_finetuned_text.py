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

class QwenFineTunedModelText(nn.Module):
    def __init__(self, cache_model = "/cluster/scratch/arsood/cache_hugging_face/Qwen2.5-VL-3B-Instruct/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3", local_files_only= True, is_training = True, path_checkpoint = "", device = 'cuda'):
        super(QwenFineTunedModelText, self).__init__()  

        self.device = device

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            cache_model, torch_dtype=torch.bfloat16, device_map=device)

        self.processor = AutoProcessor.from_pretrained(
            "/cluster/scratch/arsood/cache_hugging_face/Qwen2.5-VL-3B-Instruct/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3"
            )

        self.peft_model = self.model

        for param in self.peft_model.parameters():
            param.requires_grad = True

        for name, param in self.peft_model.named_parameters():
            if param.requires_grad:
                print(f"{name}: requires_grad = {param.requires_grad}")

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
        batch, labels = self.prepare_input_for_training(x["messages"], None, x["front_images"])
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
        batch, labels = self.prepare_input_for_training(x["messages"], None, x["front_images"])
        with torch.no_grad():
            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                outputs = self.peft_model(**batch)
                loss_value_val = self.loss(outputs.logits, labels)
        return loss_value_val.item()

    def save(self, path_to_save):
        self.peft_model.save_pretrained(path_to_save, safe_serialization=True)

    def load(self, path_checkpoint):
        model_combined = PeftModel.from_pretrained(self.model, path_checkpoint)
        model_combined.train()
        model_combined.enable_adapter_layers()
        return model_combined
    
    def generate(self, batch_input, max_new_tokens = 400):
        self.is_training = False
        input_ids = self.prepare_input_for_training(batch_input["messages"], None, batch_input["front_images"])
        outputs = self.peft_model.generate(**input_ids, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids.input_ids, outputs)
        ]
        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return output_text

    
    