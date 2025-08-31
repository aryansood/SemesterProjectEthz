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

class QwenModelVirtualTokens(nn.Module):
    def __init__(self, cache_dir = '/cluster/scratch/arsood/cache_hugging_face', local_files_only= True, new_tokens_to_add = 20, is_training = True, is_lora_config = True, device = 'cuda'):
        super(QwenModelVirtualTokens, self).__init__()

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
            r=64,
            bias="none",
            target_modules=["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",], 
            task_type="CAUSAL_LM",
        )

        self.peft_model = None
        if is_lora_config == True:
            self.peft_model = get_peft_model(self.model, peft_config)
        else:
            self.load()

        size_prev_embedding = self.peft_model.get_input_embeddings().weight.shape
        embedding_virtual = nn.Embedding(size_prev_embedding[0] + new_tokens_to_add, size_prev_embedding[1])
        with torch.no_grad():
            embedding_virtual.weight[0:size_prev_embedding[0]] = (self.peft_model.get_input_embeddings().weight.clone())
        self.peft_model.set_input_embeddings(embedding_virtual)

        # for name, param in self.peft_model.named_parameters():
        #     if param.requires_grad:
        #         print(f"{name}: requires_grad = {param.requires_grad}")

        self.virtual_tokens_list_num = torch.arange(size_prev_embedding[0], size_prev_embedding[0]+new_tokens_to_add, dtype=torch.int64)
        self.attention_mask_virtual_tokens = torch.ones(new_tokens_to_add)

        self.linear_to_traj = nn.Linear(size_prev_embedding[1], 2).to(device)

        # if torch.cuda.device_count() > 1:
        #     self.peft_model = nn.DataParallel(self.peft_model)
        self.peft_model.to(device)
        self.virtual_tokens_len = new_tokens_to_add

    def prepare_input_for_training(self, messages, images, videos):
        
        len_batch = len(messages) 
        self.processor.tokenizer.padding_side = "left"
        texts = [
        self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
        for msg in messages
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
        for i in range(0, len_batch):
            labels[i][:int(indices[1][len_token_spec*(i+1)-1])] = -100
            labels[i][-size_virtual_tokens:] = -100
        batch = batch.to(self.device)
        labels = labels.to(self.device)

        return batch, labels
        
        # output_text = self.processor.batch_decode(
        # labels, skip_special_tokens=False, clean_up_tokenization_spaces=False)



    def loss(self, logits, labels, traj_gt, traj_pred, lambda_1 = 1, lambda_2 = 1):
        logits = logits.permute(0, 2, 1)
        logits = logits[..., :-1]
        labels = labels[..., 1:]
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        loss_traj = (((traj_gt-traj_pred)**2).sum(dim=-1)).mean()
        loss_cross = criterion(logits, labels)
        loss_tot = lambda_1*loss_cross+lambda_2*loss_traj
        print("Loss cross", loss_cross)
        print("Loss traj", loss_traj)
        return loss_tot
        

    def forward(self, x):
        batch, labels = self.prepare_input_for_training(x[0], None, x[1])
        #output = self.peft_model(**batch, output_hidden_states = True)
        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            outputs = self.peft_model(**batch, output_hidden_states=True)
        last_hidden_layer_virtual_tokens = outputs.hidden_states[-1][:, -self.virtual_tokens_len:, :]
        traj_output = self.linear_to_traj(last_hidden_layer_virtual_tokens)
        loss_value = self.loss(outputs.logits, labels, torch.from_numpy(np.array(x[3])).to(self.device), traj_output)
        return outputs, loss_value
    
    def forward_inference(self, x):
        batch, labels = self.prepare_input_for_training(x[0], None, x[1])
        #output = self.peft_model(**batch, output_hidden_states = True)
        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            outputs = self.peft_model(**batch, output_hidden_states=True)
        last_hidden_layer_virtual_tokens = outputs.hidden_states[-1][:, -self.virtual_tokens_len:, :]
        traj_output = self.linear_to_traj(last_hidden_layer_virtual_tokens)
        loss_value = self.loss(outputs.logits, labels, torch.from_numpy(np.array(x[3])).to(self.device), traj_output)
        return outputs, loss_value

    def validate():
        pass

    def trainer():
        pass

    def load():
        pass

    def prepare_input_for_inference(self, messages, images, videos):
        len_batch = len(messages) 
        self.processor.tokenizer.padding_side = "left"
        texts = [
        self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
        for msg in messages
        ]
        batch = self.processor(
            text=texts,
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
        )
        virtual_token_expand = self.virtual_tokens_list_num.expand(len_batch , -1)
        attention_mask_virtual_expand = self.attention_mask_virtual_tokens.expand(len_batch , -1)
        batch["input_ids"] = torch.concat([batch["input_ids"], virtual_token_expand], dim = -1)
        batch["attention_mask"] = torch.concat([batch["attention_mask"], attention_mask_virtual_expand], dim = -1)
        
        
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
        generated_ids = self.peft_model.generate(**inputs, max_new_tokens=200)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text
    
    