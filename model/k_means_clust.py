from typing import Tuple
import os
import math
import numpy as np
import cv2
from collections import Counter
import matplotlib.pyplot as plt
from torch import nn
import torch
from tqdm import tqdm
from scipy.interpolate import CubicSpline

class KMeansGpu(nn.Module):
    def __init__(self, x_rand, y_rand):
        super(KMeansGpu, self).__init__()
        self.class_clust_x = x_rand.to('cuda')
        self.class_clust_y = y_rand.to('cuda')

    def forward(self, x : torch.Tensor, y: torch.Tensor):
        x_reshape = x.unsqueeze(1)
        y_reshape = y.unsqueeze(1)

        x_diff = (self.class_clust_x - x_reshape) ** 2
        y_diff = (self.class_clust_y - y_reshape) ** 2
        dist = torch.sqrt(x_diff + y_diff).sum(dim=2)

        min_indices = torch.argmin(dist, dim=1)
        mask = torch.nn.functional.one_hot(min_indices, num_classes=self.class_clust_x.shape[0]).float()

        mask_expand = mask.unsqueeze(2)
        sum_per_group_x = torch.sum(x_reshape * mask_expand, dim=0)
        sum_per_group_y = torch.sum(y_reshape * mask_expand, dim=0)

        counts_per_group = mask_expand.sum(dim=0)
        counts_per_group[counts_per_group == 0] = 1

        self.class_clust_x = sum_per_group_x / counts_per_group
        self.class_clust_y = sum_per_group_y / counts_per_group
        

pos_x_arr = np.load('data/traj_data/traj_fut_pos_x.npy')
pos_y_arr = np.load('data/traj_data/traj_fut_pos_y.npy')
indices = np.random.choice(pos_x_arr.shape[0], 50, replace=False)
rand_x = pos_x_arr[indices]
rand_y = pos_y_arr[indices]
rand_x = torch.from_numpy(rand_x).to('cuda')
rand_y = torch.from_numpy(rand_y).to('cuda')

pos_x_arr = torch.from_numpy(pos_x_arr).to('cuda')
pos_y_arr = torch.from_numpy(pos_y_arr).to('cuda')

kmeans = KMeansGpu(rand_x, rand_y).to('cuda')


for s in tqdm(range(1000)):
    kmeans(pos_x_arr, pos_y_arr)

x_final = kmeans.class_clust_x.detach().cpu().numpy()[:, :, None]
y_final = kmeans.class_clust_y.detach().cpu().numpy()[:, :, None]

clust_traj = np.concatenate([x_final, y_final], axis=-1)

# for i in range(0, clust_traj.shape[0]):
#     plt.plot(clust_traj[i][..., 1], clust_traj[i][..., 0], marker='o')
# plt.savefig("/cluster/home/arsood/Semester_Project_Official/plot.png",dpi=300)

np.save('data/traj_data/50_k_means_traj', clust_traj)
# np.save('/cluster/home/arsood/Semster_Project/y_k_clust_50.py', y_final)






    



    

