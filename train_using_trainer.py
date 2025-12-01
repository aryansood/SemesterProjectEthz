import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import cv2
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, get_peft_model
from trl import SFTConfig
from trl import SFTTrainer
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from dataloader_utils.dataloader_waymo import WaymoE2EDatasetTrainingAnnotated, WaymoE2EDatasetTraining
from dataloader_utils.dataloader_covla import CovlaDatasetTrainingAnnotated, CovlaDatasetTraining
from dataloader_utils.dataloader_val_waymo import WaymoE2EDatasetVal, collate_val
from dataloader_utils.collate_functions import train_collate
from torch.utils.data import DataLoader, ConcatDataset
from config import data_waymo_train, data_covla, data_waymo_val
from tqdm import tqdm
from torch.utils.data import random_split


def train():
    training_covla_annotated = CovlaDatasetTrainingAnnotated(data_covla.video, data_covla.state, 1, is_fut_traj=True)
    training_waymo_annotated = WaymoE2EDatasetTrainingAnnotated(data_waymo_train.data, 1, is_fut_traj = True)
    training_data = ConcatDataset([training_covla_annotated])

    dataset_size = len(training_data)
    val_size = int(0.1 * dataset_size)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(training_data, [train_size, val_size])

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "/cluster/scratch/arsood/qwen_3b_param_model", dtype=torch.float16, device_map='cuda')
    processor = AutoProcessor.from_pretrained( "/cluster/scratch/arsood/qwen_3b_param_model")
    
    training_args = SFTConfig(
        output_dir="/cluster/scratch/arsood/Qwen3vlm_fine_tune",  # Directory to save the model
        num_train_epochs=2,  # Number of training epochs
        per_device_train_batch_size=2,  # Batch size for training
        per_device_eval_batch_size=0,  # Batch size for evaluation
        gradient_accumulation_steps=2,  # Steps to accumulate gradients
        gradient_checkpointing=False,  # Enable gradient checkpointing for memory efficiency
        # Optimizer and scheduler settings
        optim="adamw_torch_fused",  # Optimizer type
        learning_rate=2e-4,  # Learning rate for training
        lr_scheduler_type="constant",  # Type of learning rate scheduler
        # Logging and evaluation
        logging_steps=10,  # Steps interval for logging
        save_strategy="steps",  # Strategy for saving the model
        save_steps=10,  # Steps interval for saving
        greater_is_better=False,  # Whether higher metric values are better
        # Mixed precision and gradient settings
        bf16=True,  # Use bfloat16 precision
        tf32=True,  # Use TensorFloat-32 precision
        max_grad_norm=0.3,  # Maximum norm for gradient clipping
        warmup_ratio=0.03,  # Ratio of total steps for warmup
        dataset_text_field="",  # Text field in dataset
        #report_to=["tensorboard"],
        dataset_kwargs={"skip_prepare_dataset": True},  # Additional dataset options
    )

    training_args.remove_unused_columns = False  # Keep unused columns in dataset

    def collate_fn(messages):
        len_batch = len(messages)
        videos = messages['front_images']
        processor.tokenizer.padding_side = "left"
        texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt= False) for msg in messages
        ]
        batch = processor(
            text=texts,
            images=None,
            videos=videos,
            padding=True,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        assistant_start_ids = processor.tokenizer.convert_tokens_to_ids('<|im_start|>')
        indices = torch.where(labels == assistant_start_ids)
        len_token_spec = int((indices[0].shape[0])/len_batch)
        for i in range(0, len_batch):
            labels[i][:int(indices[1][len_token_spec*(i+1)-1])] = -100
        
        batch["labels"] = labels
        
        batch = batch.to('cuda')

        return batch
        
    
    trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=None,
    data_collator=collate_fn,
    processing_class=processor.tokenizer,
    )
    trainer.train()

if __name__ == "__main__":
    train()