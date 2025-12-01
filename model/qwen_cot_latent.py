import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import cv2
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from torch import nn
from torch.optim import Adam
from scipy.interpolate import CubicSpline
import re
import json

class QwenLatentCot(nn.Module):
    def __init__(self, cache_model, is_training = True, device = 'cuda', new_tokens_to_add = 20):
        super(QwenLatentCot, self).__init__()

        self.device = device

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            cache_model, torch_dtype=torch.bfloat16, device_map=device)

        self.processor = AutoProcessor.from_pretrained(
            cache_model
            )

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.to(device)
        self.is_training = is_training

        size_prev_embedding = self.model.get_input_embeddings().weight.shape
        embedding_virtual = nn.Embedding(size_prev_embedding[0] + new_tokens_to_add, size_prev_embedding[1])
        with torch.no_grad():
            embedding_virtual.weight[0:size_prev_embedding[0]] = (self.model.get_input_embeddings().weight.clone())
        self.model.set_input_embeddings(embedding_virtual)
        self.virtual_tokens_list_num = torch.arange(size_prev_embedding[0], size_prev_embedding[0]+new_tokens_to_add, dtype=torch.int64)
        self.attention_mask_virtual_tokens = torch.ones(new_tokens_to_add)

        self.linear_to_traj = nn.Linear(size_prev_embedding[1], 2).to(device)
        self.model.to(device)
        self.virtual_tokens_len = new_tokens_to_add

    def prepare_input_for_training(self, messages, images, videos):
        
        len_batch = len(messages) 
        self.processor.tokenizer.padding_side = "left"
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt= True) for msg in messages
        ]
        batch = self.processor(
            text=texts,
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
        )
        size_virtual_tokens = self.attention_mask_virtual_tokens.shape[0]
        virtual_token_expand = self.virtual_tokens_list_num.expand(len_batch , -1)
        attention_mask_virtual_expand = self.attention_mask_virtual_tokens.expand(len_batch , -1)
        batch["input_ids"] = torch.concat([batch["input_ids"], virtual_token_expand], dim = -1)
        batch["attention_mask"] = torch.concat([batch["attention_mask"], attention_mask_virtual_expand], dim = -1)
        labels = batch["input_ids"].clone()
        assistant_start_ids = self.processor.tokenizer.convert_tokens_to_ids('<|im_start|>')
        indices = torch.where(labels == assistant_start_ids)
        len_token_spec = int((indices[0].shape[0])/len_batch)
        batch = batch.to(self.device)
        return batch

    def forward(self, x):
        batch = self.prepare_input_for_training(x["messages"], None, x["front_images"])
        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            outputs = self.model(**batch, output_hidden_states=True)
        last_hidden_layer_virtual_tokens = outputs.hidden_states[-1][:, -self.virtual_tokens_len:, :]
        print(last_hidden_layer_virtual_tokens.shape)
        return outputs
    
    def loss_validation(self, pred, target):
        loss_diff = torch.sqrt(torch.square(pred-target).sum(dim = -1))
        return loss_diff

    def validate(self, x):
        batch, labels = self.prepare_input_for_training(x["messages"], None, x["front_images"])
        with torch.no_grad():
            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                outputs = self.model(**batch)
                loss_value_val = self.loss(outputs.logits, labels)
        return loss_value_val.item()

    def save(self, path_to_save):
        self.model.save_pretrained(path_to_save, safe_serialization=True)