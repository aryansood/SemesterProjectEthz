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
from torchvision import transforms
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

class QwenEncoder(nn.Module):
    def __init__(self, cache_model = "/cluster/scratch/arsood/cache_hugging_face/Qwen2.5-VL-3B-Instruct/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3", local_files_only= True, path_checkpoint = "", device = 'cuda'):
        super(QwenEncoder, self).__init__()

        self.device = device

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            cache_model, torch_dtype=torch.bfloat16, device_map=device)

        self.processor = AutoProcessor.from_pretrained(
            "/cluster/scratch/arsood/cache_hugging_face/Qwen2.5-VL-3B-Instruct/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3"
            )
        
        for name, module in self.model.named_modules():
            print(name)
        
    def prepare_input_generation(self, messages, images, videos):
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
        batch = batch.to(self.device)
        return batch

    def forward(self, batch_input, max_new_tokens = 400):
        input_ids = self.prepare_input_generation(batch_input["messages"], None, batch_input["front_images"])
        outputs = self.model.generate(**input_ids, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids.input_ids, outputs)
        ]
        input_ids["input_ids"] = outputs
        exstra_ones = torch.ones(outputs.shape[0], outputs.shape[1]-input_ids["attention_mask"].shape[1]).to('cuda')
        input_ids["attention_mask"] = torch.concat([input_ids["attention_mask"], exstra_ones], dim = 1)
        out_2 = self.model(**input_ids, output_hidden_states=True, return_dict=True)

        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return output_text
    
    def post_processing_model(self, batch):
        self.processor.tokenizer.padding_side = "right"
        tokenized_batch = self.processor.tokenizer(batch, return_tensors="pt", padding=True)['input_ids'].to('cuda')
        out_embedd = self.model.get_input_embeddings()(tokenized_batch)

class DinoEncoder(nn.Module):
    def __init__(self, dino3_dir = "/cluster/scratch/arsood/DinoV3", device = 'cuda'):
        super(DinoEncoder, self).__init__()
        self.device = device
        self.model = torch.hub.load("/cluster/home/arsood/dinov3/dinov3", 'dinov3_vith16plus', source='local', weights='/cluster/home/arsood/dinov3/checkpoint/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth')

    def forward(self, batch):
        transform_dino = self.make_transform(512)
        image_tran = transform_dino(np.array(batch["front_images"]))
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
    
    def forward(self, batch_in, batch_mem, memory_key_padding_mask):
        out = self.transformer_decoder(batch_in, batch_mem, memory_key_padding_mask=memory_key_padding_mask)
        return out

class DinoQwenDiffusion(nn.Module):
    def __init__(self, device='cuda', k_means_traj_loc = "data/traj_data/50_k_means_traj.npy", beta_start=1e-4, beta_end = 0.1, Time_den = 100, text_dim_in = 2048, traj_in_dim = 20, traj_hidden_dim = 512, mem_out_dim = 1024):
        super(DinoQwenDiffusion, self).__init__()
        self.device = device
        self.k_mean_anchored = np.load(k_means_traj_loc)
        beta_step = np.linspace(beta_start, beta_end, Time_den)
        self.alpha_step = 1 - beta_step
        self.alpha_bar = np.cumprod(self.alpha_step)
        self.k_mean_anchored = self.k_mean_anchored.reshape(self.k_mean_anchored.shape[0], self.k_mean_anchored.shape[1]*self.k_mean_anchored.shape[2])
        self.transformer_decoder = TrajDecoderDiff()
        self.ln_text_encoder = nn.Linear(text_dim_in, mem_out_dim)
        self.mpl_status_past_traj = nn.Sequential(
            nn.Linear(traj_in_dim, traj_hidden_dim),
            nn.ReLU(),
            nn.Linear(traj_hidden_dim, traj_hidden_dim),
            nn.ReLU(),
            nn.Linear(traj_hidden_dim, mem_out_dim)
        )
        self.pos_enc = self.PositionalEncoding()

    def forward(self, batch_input, batch_memory, time_step, num_traj):
        shape_memory = batch_memory.shape
        batch_memory = batch_memory.unsqueeze(1).expand(shape_memory[0], num_traj, shape_memory[1], shape_memory[2])
        batch_memory = batch_memory.reshape(shape_memory[0]*num_traj, shape_memory[1], shape_memory[2])

        batch_input = batch_input.reshape(batch_input.shape[0]*batch_input[1], batch_input[2], batch_input[3])
        batch_input = self.pos_enc(batch_input)

        out_traj = self.transformer_decoder(batch_input, batch_memory)


    def sample_traj_init(self, num_traj, denoising_step, Time_den = 100):
        time_step_sample = [1]#random.sample(range(0, Time_den), 1)
        list_index_k_means = random.sample(range(0, self.k_mean_anchored.shape[0]), num_traj)
        sample_traj = self.k_mean_anchored#[list_index_k_means]
        noise_t = torch.randn(sample_traj.shape[0], sample_traj.shape[1]).numpy()
        noise_time_step_t = np.sqrt(self.alpha_bar[time_step_sample[0]])*sample_traj+np.sqrt(1-self.alpha_bar[time_step_sample[0]])*noise_t
        noise_time_step_t = noise_time_step_t.reshape(noise_time_step_t.shape[0], 20, 2)
        # plt.figure()
        # plt.xlim([-20, 20])
        # for i in range(0, noise_time_step_t.shape[0]):
        #     plt.plot(noise_time_step_t[i][..., 1], noise_time_step_t[i][..., 0])
        # plt.savefig("/cluster/home/arsood/Semester_Project_Official/noise_traj.png")





if __name__ == "__main__":
    dino_obj = DinoQwenDiffusion()
    dino_obj.sample_traj_init(10, 9)







