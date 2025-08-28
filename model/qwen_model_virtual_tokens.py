import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import cv2
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model
from torch import nn

class QwenModelVirtualTokens(nn.Module):
    def __init__(self, cache_dir = '/cluster/scratch/arsood/cache_hugging_face', local_files_only= True, new_tokens_to_add = 20, is_training = True, is_lora_config = True, device = 'cuda'):

        model_qwen_select =  "Qwen/Qwen2.5-VL-3B-Instruct"
        
        self.device = device

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_qwen_select, torch_dtype=torch.float16, device_map=device,
            cache_dir = cache_dir, local_files_only= local_files_only)

        self.processor = AutoProcessor.from_pretrained(
            model_qwen_select,
            cache_dir=cache_dir,
            local_files_only=local_files_only
            )
        
        self.linear_to_traj = nn.Linear(2048, 20)

        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.05,
            r=16,
            bias="none",
            target_modules=["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",], 
            task_type="CAUSAL_LM",
        )

        embedding_virtual = nn.Embedding(V + n_new, D)

        print(self.model.embeddings.weight.shape[0])
        len_new_tokens = self.model.embeddings.weight.shape[0]+new_tokens_to_add
        self.peft_model.resize_token_embeddings(len_new_tokens)

        self.peft_model = None
        if is_lora_config == True:
            self.peft_model = get_peft_model(self.model, peft_config)
        else:
            self.load()

        
        
    
    def prepare_input_for_training(self, messages, images, videos):
        self.processor.tokenizer.padding_side = "left"
        texts = [
        self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
        ]
        batch = self.processor(
            text=texts,
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
        )
    
        batch = batch.to('cuda')
        labels = batch["input_ids"].clone()
        assistant_start_ids = self.processor.tokenizer.convert_tokens_to_ids('<|im_start|>')
        indices = torch.where(labels == assistant_start_ids)
        indices_2 = indices[1][2::3]
        for i in range(0, indices_2.shape[0]):
            labels[i][:indices_2[i]] = -100


    def loss(logits, labels):
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        loss = criterion(logits, labels)
        return loss
        

    def forward(self, x, labels):
        self.peft_model()
        
        pass

    def validate():
        pass

    def trainer():
        pass

    def load():
        pass

    def prepare_input_for_inference():
        pass
        
    def generate(self, messages, images, videos):
        """
        messages: List of messages-> Depends on the batch
        images: List of Images-> Put all the Images in the order they appear
        videos: List of Videos-> Put all the Videos in the order they appear
        """
        self.processor.tokenizer.padding_side = "left"
        texts = [
        self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
        ]
        inputs = self.processor(
            text=texts,
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
        )
    
        inputs = inputs.to('cuda')
        generated_ids = self.model.generate(**inputs, max_new_tokens=300)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text
    
    