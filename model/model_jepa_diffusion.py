import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torch import nn
from torch.optim import Adam
from transformers import AutoVideoProcessor
from transformers import CLIPModel, CLIPProcessor

class JepaPredictTraj(nn.Module):
    def __init__(self, device):
        super(JepaPredictTraj, self).__init__()

        self.device = device

        self.back_bone = AutoModel.from_pretrained(
        "facebook/vjepa2-vitl-fpc64-256",
        dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa"
        )
        processor = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitl-fpc64-256")

        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        """
        inputs = processor(text=texts, return_tensors="pt", padding=True)

        # Encode the text with CLIP text encoder
        with torch.no_grad():
            output = model.get_text_features(**inputs) 
        """
        self.mlp_clip_embedding = nn.Linear()
        self.mpl_ergo_status = nn.Linear()




    def forward(self, )

