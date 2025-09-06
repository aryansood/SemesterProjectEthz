import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torch import nn
from torch.optim import Adam
from transformers import AutoVideoProcessor

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

        #model.get_vision_features()`








    def forward(self, )

