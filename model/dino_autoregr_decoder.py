import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torch import nn
from torch.optim import Adam
from transformers import AutoVideoProcessor, AutoImageProcessor
from transformers import AutoModel
import matplotlib.pyplot as plt
from torchvision  import transforms
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import random
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        #x is [B, seq, embeddings]
        x = x.permute(1,0,2)
        x = x + self.pe[:x.size(0)]
        x = x.permute(1,0,2)
        return x

class DinoEncoder(nn.Module):
    def __init__(self, dino3_dir = "/cluster/scratch/arsood/DinoV3", device = 'cuda'):
        super(DinoEncoder, self).__init__()
        self.device = device
        self.model = torch.hub.load("/cluster/home/arsood/dinov3/dinov3", 'dinov3_vith16plus', source='local', weights='/cluster/home/arsood/dinov3/checkpoint/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth')
        self.model.to(device)

    def forward(self, batch):
        transform_dino = self.make_transform(512)
        image_tran = [transform_dino(np.array(batch["front_images_list_no"][i][-1].cpu())) for i in range(0, len(batch["front_images_list_no"]))]#transform_dino(np.array(batch["front_images_no"])[:, -1, :, : , :])
        image_tran = torch.stack(image_tran).to(self.device)
        video_features = self.model(image_tran, is_training=True)['x_norm_patchtokens']
        return video_features

    @staticmethod
    def make_transform(resize_size: int | tuple[int, int] = 768):
        resize = transforms.Resize((resize_size, resize_size), antialias=True)
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        return transforms.Compose([to_tensor, resize, normalize])
    

class DinoAutoregrDecoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, intent_emb_dim = 4, traj_input_dim = 2):
        super(DinoAutoregrDecoder, self).__init__()
        transformer_layer = nn.TransformerDecoderLayer(d_model= d_model, nhead = num_heads, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(transformer_layer, num_layers= num_layers)
        self.intent_emb = nn.Embedding(intent_emb_dim, d_model)
        self.input_traj_emb = nn.Linear(traj_input_dim, d_model)
        self.out_traj_emb = nn.Linear(d_model, traj_input_dim)
        self.positional_encoding = PositionalEncoding(d_model)
        
    def forward(self, batch_vis_feat, batch_intent, batch_input_traj):
        intent_emb_feat = self.intent_emb(batch_intent)
        intent_emb_feat = intent_emb_feat.unsqueeze(dim = 1)
        mem_concat_feat = torch.concat([batch_vis_feat, intent_emb_feat], dim = 1)
        input_traj_emb_feat = self.input_traj_emb(batch_input_traj)
        input_traj_emb_feat = self.positional_encoding(input_traj_emb_feat)
        mask_tgt = nn.Transformer.generate_square_subsequent_mask(21)
        out_traj_dim = self.transformer_decoder(input_traj_emb_feat, mem_concat_feat, tgt_mask=mask_tgt)
        out_final_traj = self.out_traj_emb(out_traj_dim)
        return out_final_traj[:, 1:, :]

class PipiLineDinoAutoregr(nn.Module):
    def loss_ade(self, target_traj, predict_traj):
        loss = torch.square(target_traj-predict_traj).sum(dim = -1).sqrt().mean()
        return loss

    def __init__(self, d_model = 1280, num_heads = 4, num_layers = 8, intent_emb_dim = 4, traj_input_dim = 2, device = 'cuda'):
        super(PipiLineDinoAutoregr, self).__init__()
        self.dino_auto = DinoAutoregrDecoder(d_model = d_model, num_heads = num_heads, num_layers = num_layers, intent_emb_dim = intent_emb_dim, traj_input_dim = traj_input_dim)
        self.dino_encoder = DinoEncoder()
        self.dino_auto.to(device)
        self.dino_encoder.to(device)
        self.device = device
        for param in self.dino_encoder.parameters():
            param.requires_grad = False


    def forward(self, batch):
        with torch.no_grad():
            visual_feat = self.dino_encoder(batch)
        vel_vec = torch.from_numpy(np.array(batch["cur_vel"])).float().to(self.device).unsqueeze(1)
        emb_feat = torch.from_numpy(np.array(batch["num_intent"])).to(self.device)
        emb_traj_input = torch.from_numpy(np.array(batch["next_state_traj"])).float().to(self.device)
        emb_input_decoder = torch.concat([vel_vec, emb_traj_input], dim = 1).float().to(self.device)
        fut_traj_gt = torch.from_numpy(np.array(batch["next_state_traj"])).to(self.device)
        out = self.dino_auto(visual_feat, emb_feat, emb_input_decoder)
        loss_ade_value = self.loss_ade(out, fut_traj_gt)
        return out, loss_ade_value
    
    def validate(self, batch):
        with torch.no_grad():
            visual_feat = self.dino_encoder(batch)
            vel_vec = torch.from_numpy(np.array(batch["cur_vel"])).float().to(self.device).unsqueeze(1)
            emb_feat = torch.from_numpy(np.array(batch["num_intent"])).to(self.device)
            emb_traj_input = torch.from_numpy(np.array(batch["next_state_traj"])).float().to(self.device)
            emb_input_decoder = torch.concat([vel_vec, emb_traj_input], dim = 1).float().to(self.device)
            fut_traj_gt = torch.from_numpy(np.array(batch["next_state_traj"])).to(self.device)
            out = self.dino_auto(visual_feat, emb_feat, emb_input_decoder)
            loss_ade_value = self.loss_ade(out, fut_traj_gt)
            loss_ade_value = loss_ade_value.detach().cpu()
        return loss_ade_value
    
    def save(self, path_to_save):
        torch.save(self.state_dict(), path_to_save)      
        








        

        
