import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import cv2
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from torch import nn
from scipy.interpolate import CubicSpline
#Prev was 300 Latents
class QwenLatentCot(nn.Module):
    def __init__(self, cache_model, processor_config ,is_training = True, device = 'cuda', cot_latent_num_tokens = 300, latent_traj_tokens = 6):
        super(QwenLatentCot, self).__init__()

        self.device = device

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            cache_model, torch_dtype=torch.float32, device_map=device)

        self.processor = AutoProcessor.from_pretrained(
            processor_config
            )

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.to(device)
        self.is_training = is_training

        size_prev_embedding = self.model.get_input_embeddings().weight.shape
        self.temp_size = size_prev_embedding[0]
        self.embedding_virtual = nn.Embedding(size_prev_embedding[0] + cot_latent_num_tokens, size_prev_embedding[1])
        with torch.no_grad():
            self.embedding_virtual.weight[0:size_prev_embedding[0]] = (self.model.get_input_embeddings().weight.clone())

        self.mask_grad = torch.zeros(self.embedding_virtual.weight.shape[0], self.embedding_virtual.weight.shape[1], dtype=torch.bool)
        self.mask_grad[-cot_latent_num_tokens:] = True

        self.model.set_input_embeddings(self.embedding_virtual)
        self.virtual_tokens_list_num = torch.arange(size_prev_embedding[0], size_prev_embedding[0]+cot_latent_num_tokens, dtype=torch.int64)
        self.attention_mask_virtual_tokens = torch.ones(cot_latent_num_tokens)

        self.hidden_to_traj = nn.Linear(size_prev_embedding[1], 2).to(device)
        self.model.to(device)
        self.latent_traj_tokens_len = latent_traj_tokens
        self.cot_latent_num_tokens = cot_latent_num_tokens

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
        virtual_token_expand = self.virtual_tokens_list_num.expand(len_batch , -1)
        attention_mask_virtual_expand = self.attention_mask_virtual_tokens.expand(len_batch , -1)
        batch["input_ids"] = torch.concat([batch["input_ids"], virtual_token_expand], dim = -1)
        batch["attention_mask"] = torch.concat([batch["attention_mask"], attention_mask_virtual_expand], dim = -1)
        batch = batch.to(self.device)
        return batch

    def forward(self, x):
        batch = self.prepare_input_for_training(x["messages"], None, x["front_images"])
        target = torch.tensor(np.array(x["next_state_traj"])).to(self.device)
        index = [0, 3, 7, 11, 15, 19]
        target = target[:, index]
        with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            outputs = self.model(**batch, output_hidden_states=True)
        last_hidden_layer_virtual_tokens = outputs.hidden_states[-1][:, -self.latent_traj_tokens_len:, :]
        pred = self.hidden_to_traj(last_hidden_layer_virtual_tokens)
        loss_to_return = self.loss(pred, target)
        return pred, loss_to_return
    
    def loss(self, pred, target):
        loss_to_return = torch.mean(torch.sqrt(torch.square(pred-target).sum(dim = -1)))
        return loss_to_return
    
    def loss_validation(self, pred, target):
        loss_diff = torch.sqrt(torch.square(pred-target).sum(dim = -1))
        return loss_diff

    def validate(self, x):
        batch = self.prepare_input_for_training(x["messages"], None, x["front_images"])
        target = torch.tensor(np.array(x["next_state_traj"])).to(self.device)
        # index = [0, 3, 7, 11, 15, 19]
        # target = target[:, index]
        with torch.no_grad():
            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                outputs = self.model(**batch, output_hidden_states=True)
                # token_ids = outputs['logits'][:, -self.cot_latent_num_tokens:, :].argmax(dim=-1)
                # output_text = self.processor.batch_decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                # print(output_text)
            last_hidden_layer_virtual_tokens = outputs.hidden_states[-1][:, -self.latent_traj_tokens_len:, :]
            pred = self.hidden_to_traj(last_hidden_layer_virtual_tokens)
            pred_arr_20_hz = []
            for i in range(0, pred.shape[0]):
                index = [0, 3, 7, 11, 15, 19]
                traj_pred_to_consider = pred[i].clone().detach().cpu().numpy()
                cs_x = CubicSpline(index, traj_pred_to_consider[...,0])
                cs_y = CubicSpline(index, traj_pred_to_consider[...,1])
                t_high = np.arange(0, 20)
                x_high = cs_x(t_high)[:, None]
                y_high = cs_y(t_high)[:, None]
                traj_pred = np.concatenate([x_high, y_high], axis = -1)
                pred_arr_20_hz.append(traj_pred)
            pred_arr_20_hz = np.array(pred_arr_20_hz)
            pred_arr_20_hz = torch.from_numpy(pred_arr_20_hz)
            pred_arr_20_hz = pred_arr_20_hz.to(self.device)
            loss_to_return = self.loss_validation(pred_arr_20_hz, target)
        return loss_to_return#, pred_arr_20_hz.detach().cpu().numpy()
    
    def generate_trajectory(self, x):
        batch = self.prepare_input_for_training(x["messages"], None, x["front_images"])
        with torch.no_grad():
            with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                outputs = self.model(**batch, output_hidden_states=True)
            last_hidden_layer_virtual_tokens = outputs.hidden_states[-1][:, -self.latent_traj_tokens_len:, :]
            pred = self.hidden_to_traj(last_hidden_layer_virtual_tokens)
            pred_arr_20_hz = []
            for i in range(0, pred.shape[0]):
                index = [0, 3, 7, 11, 15, 19]
                traj_pred_to_consider = pred[i].clone().detach().cpu().numpy()
                cs_x = CubicSpline(index, traj_pred_to_consider[...,0])
                cs_y = CubicSpline(index, traj_pred_to_consider[...,1])
                t_high = np.arange(0, 20)
                x_high = cs_x(t_high)[:, None]
                y_high = cs_y(t_high)[:, None]
                traj_pred = np.concatenate([x_high, y_high], axis = -1)
                pred_arr_20_hz.append(traj_pred)
            pred_arr_20_hz = np.array(pred_arr_20_hz)
            pred_arr_20_hz = torch.from_numpy(pred_arr_20_hz)
            pred_arr_20_hz = pred_arr_20_hz.to(self.device)
        return pred_arr_20_hz.detach().cpu().numpy()

    def save(self, path_to_save_virt, path_to_save_linear):
        #weight_to_save = self.model.get_input_embeddings().weight#.weight[-self.cot_latent_num_tokens:].clone()
        torch.save(self.model.get_input_embeddings().weight, path_to_save_virt)
        torch.save(self.hidden_to_traj.state_dict(), path_to_save_linear)
    
    def save_all(self, path_to_save):
        torch.save(self, path_to_save)
        
    def load_model(self, path_from_to_load_virt, path_from_to_load_linear):
        tensor_to_load = torch.load(path_from_to_load_virt)
        tensor_to_load.to(self.device)
        with torch.no_grad():
            self.model.get_input_embeddings().weight = tensor_to_load
        self.hidden_to_traj.load_state_dict(torch.load(path_from_to_load_linear))

    def generate(self, batch_input, max_new_tokens = 400):
        input_ids = self.prepare_input_for_training(batch_input["messages"], None, batch_input["front_images"])
        outputs = self.model.generate(**input_ids, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids.input_ids, outputs)
        ]
        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return output_text  