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


class TrajDecoderDiff(nn.Module):
    def __init__(self, dim_model, num_attention_head, num_layer):
        super(TrajDecoderDiff, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model=dim_model, nhead=num_attention_head, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layer)
    
    def forward(self, batch_in, batch_mem):
        out = self.transformer_decoder(batch_in, batch_mem)
        return out

class Traj_Decoder(nn.Module):
    def __init__(self, device='cuda', traj_in_dim = 20, traj_hidden_dim = 512, dim_tran = 1024, num_head = 4, num_layer = 8):
        super(Traj_Decoder, self).__init__()
        self.device = device
        self.transformer_decoder = TrajDecoderDiff(dim_tran, num_head , num_layer)
        self.ln_status_past_traj = nn.Linear(2, dim_tran)
        self.pos_enc = PositionalEncoding(dim_tran)
        self.virtual_tokens = nn.Parameter(torch.randn(20, dim_tran))
        self.intent_emb = nn.Embedding(4, dim_tran)
        self.last_emb = nn.Linear(dim_tran, 2)

    def forward(self, batch_memory, batch_past_ego, batch_intent):
        batch_size = batch_memory.shape[0]
        x = self.virtual_tokens.unsqueeze(0)
        x = self.pos_enc(x)
        x = x.expand(batch_size, -1, -1)
        ego_past_traj = self.ln_status_past_traj(batch_past_ego)
        ego_intent_traj = self.intent_emb(batch_intent)
        features = torch.concat([batch_memory, ego_past_traj, ego_intent_traj], dim = 1)
        hidden_layer = self.transformer_decoder(x, features)
        out = self.last_emb(hidden_layer)
        print(self.virtual_tokens[0])
        return out

class PipelineDino(nn.Module):
    def __init__(self):
        super(PipelineDino, self).__init__()
        self.dino_encoder = DinoEncoder()
        self.dino_traj_decoder = Traj_Decoder()
    
    def forward(self, batch):
        with torch.no_grad():
            visual_feature = self.dino_encoder(batch)
        ego_feature = torch.tensor(np.array(batch["past_state_traj"]), dtype=torch.float32).to('cuda')
        ego_intent = torch.tensor(np.array(batch["index_intent"])).unsqueeze(dim = 1).to('cuda')
        out = self.dino_traj_decoder(visual_feature, ego_feature, ego_intent).to('cuda')
        predict_traj = torch.tensor(np.array(batch["next_state_traj"])).to('cuda')
        # indices = [0, 3, 7, 11, 15, 19]
        # predict_traj = predict_traj[:, indices]
        loss = self.loss_ade(out, predict_traj)
        return out, loss

    def loss_ade(self, target_traj, predict_traj):
        loss = torch.square(target_traj-predict_traj).sum(dim = -1).sqrt().mean()
        return loss
    
    def validate(self, batch):
        with torch.no_grad():
            predict_traj, loss = self.forward(batch)
        return loss

    
        




# class QwenEncoder(nn.Module):
#     def __init__(self, cache_model = "/cluster/scratch/arsood/cache_hugging_face/Qwen2.5-VL-3B-Instruct/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3", local_files_only= True, path_checkpoint = "", device = 'cuda'):
#         super(QwenEncoder, self).__init__()

#         self.device = device

#         self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#             cache_model, torch_dtype=torch.bfloat16, device_map=device)

#         self.processor = AutoProcessor.from_pretrained(
#             "/cluster/scratch/arsood/cache_hugging_face/Qwen2.5-VL-3B-Instruct/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3"
#             )
                
#     def prepare_input_generation(self, messages, images, videos):
#         self.processor.tokenizer.padding_side = "left"
#         texts = [
#             self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt= True) for msg in messages
#         ]
#         batch = self.processor(
#             text=texts,
#             images=images,
#             videos=videos,
#             padding=True,
#             return_tensors="pt",
#         )
#         batch = batch.to(self.device)
#         return batch

#     def forward(self, batch_input, max_new_tokens = 300):
#         input_ids = self.prepare_input_generation(batch_input["messages"], None, batch_input["front_images"])
#         outputs = self.model.generate(**input_ids, max_new_tokens=max_new_tokens)
#         generated_ids_trimmed = [
#             out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids.input_ids, outputs)
#         ]
#         # input_ids["input_ids"] = outputs
#         # exstra_ones = torch.ones(outputs.shape[0], outputs.shape[1]-input_ids["attention_mask"].shape[1]).to('cuda')
#         # input_ids["attention_mask"] = torch.concat([input_ids["attention_mask"], exstra_ones], dim = 1)
#         # out_2 = self.model(**input_ids, output_hidden_states=True, return_dict=True)
#         output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
#         return output_text







