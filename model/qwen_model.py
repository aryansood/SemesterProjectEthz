import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import cv2
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model

class QwenModel():
    def __init__(self, cache_dir = '/cluster/scratch/arsood/cache_hugging_face', local_files_only= True, model_to_use = 7, device = 'cuda'):
        model_dict = {
           3: "Qwen/Qwen2.5-VL-3B-Instruct",
           7: "Qwen/Qwen2.5-VL-7B-Instruct",
           32: "Qwen/Qwen2.5-VL-32B-Instruct",
           72: "Qwen/Qwen2.5-VL-72B-Instruct"
        }
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_dict[model_to_use], torch_dtype=torch.float16, device_map=device,
            cache_dir = cache_dir, local_files_only= local_files_only)

        self.processor = AutoProcessor.from_pretrained(
            model_dict[model_to_use],
            cache_dir=cache_dir,
            local_files_only=local_files_only
            )
        self.device = device
        
    def generate(self, messages, images, videos):
        """
        messages: List of messages-> Depends on the batch
        images: List of Images-> Put all the Images in the order the appear
        videos: List of Videos-> Put all the Videos in the order the appear
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