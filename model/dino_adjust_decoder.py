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
        self.model = torch.hub.load("/cluster/home/arsood/dinov3/dinov3", 'dinov3_vitl16', source='local', weights='/cluster/home/arsood/dinov3/checkpoint/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth')
        self.model.to(device)

    def forward(self, batch):
        transform_dino = self.make_transform(512)
        image_tran = [transform_dino(np.array(batch["front_images_no"][i][-1])) for i in range(0, len(batch["front_images_no"]))]#transform_dino(np.array(batch["front_images_no"])[:, -1, :, : , :])
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
    

class DinoAdjustDecoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, intent_emb_dim = 4, traj_input_dim = 2, device = 'cuda'):
        super(DinoAdjustDecoder, self).__init__()
        transformer_layer = nn.TransformerDecoderLayer(d_model= d_model, nheads = num_heads)
        self.transformer_decoder = nn.TransformerDecoder(transformer_layer, num_layers= num_layers)
        self.intent_emb = nn.Embedding(intent_emb_dim, d_model)
        self.input_traj_emb = nn.Linear(traj_input_dim, d_model)
        self.out_traj_emb = nn.Linear(d_model, traj_input_dim)
        
    def forward(self, batch_vis_feat, batch_intent, batch_input_traj):
        intent_emb_feat = self.intent_emb(batch_intent)
        intent_emb_feat = intent_emb_feat.unsqueeze(dim = 1)
        mem_concat_feat = torch.concat([batch_vis_feat, intent_emb_feat], dim = 1)
        input_traj_emb_feat = self.input_traj_emb(batch_input_traj)
        out_traj_dim =  self.transformer_decoder(input_traj_emb_feat, mem_concat_feat)
        out_final_traj = self.out_traj_emb(out_traj_dim)+batch_input_traj

        return out_final_traj

class PipiLineDinoAdjust(nn.Module):
    def __init__(self, d_model = 1024, num_heads = 4, num_layers = 8, intent_emb_dim = 4, traj_input_dim = 2):
        super(PipiLineDinoAdjust, self).__init__()
        self.dino_auto = DinoAdjustDecoder(d_model = d_model, num_heads = num_heads, num_layers = num_layers, intent_emb_dim = intent_emb_dim, traj_input_dim = traj_input_dim)
        self.dino_encoder = DinoEncoder()

    def forward(self, batch):
        visual_feat = self.dino_encoder(batch)
        emb_feat = batch["index_intent"]
        # emb_traj_input = 


        

        
