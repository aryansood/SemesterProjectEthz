from dataloader_utils.dataloader_waymo import WaymoE2EDatasetTrainingAnnotated, WaymoE2EDatasetTraining
from dataloader_utils.dataloader_covla import CovlaDatasetTrainingAnnotated, CovlaDatasetTraining
from dataloader_utils.dataloader_val_waymo import WaymoE2EDatasetVal, collate_val
from dataloader_utils.collate_functions import train_collate
from torch.utils.data import DataLoader, ConcatDataset
from config import data_waymo_train, data_covla, data_waymo_val
from tqdm import tqdm
import numpy as np
import torch
import os
from train.train_distributed import trainer
from model.qwen_finetuned_text import QwenFineTunedModelText
from torch.utils.data import random_split
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
import matplotlib.pyplot as plt
import statistics
import tensorflow as tf

from waymo_open_dataset.wdl_limited.camera.ops import py_camera_model_ops
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2
import cv2

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer


def project_vehicle_to_image(vehicle_pose, calibration, points):
  pose_matrix = np.array(vehicle_pose.transform).reshape(4, 4)
  world_points = np.zeros_like(points)
  for i, point in enumerate(points):
    cx, cy, cz, _ = np.matmul(pose_matrix, [*point, 1])
    world_points[i] = (cx, cy, cz)
  extrinsic = tf.reshape(
      tf.constant(list(calibration.extrinsic.transform), dtype=tf.float32),
      [4, 4])
  intrinsic = tf.constant(list(calibration.intrinsic), dtype=tf.float32)
  metadata = tf.constant([
      calibration.width,
      calibration.height,
      open_dataset.CameraCalibration.GLOBAL_SHUTTER,
  ],
                         dtype=tf.int32)
  camera_image_metadata = list(vehicle_pose.transform) + [0.0] * 10

  return py_camera_model_ops.world_to_image(extrinsic, intrinsic, metadata,
                                            camera_image_metadata,
                                            world_points).numpy()

def draw_points_on_image(image, points, size, colour):
  for point in points:
    cv2.circle(image, (int(point[0]), int(point[1])), size, colour, -1)
  return image


val_dataset = WaymoE2EDatasetVal(data_waymo_val.data, 1)
val_loader = DataLoader(val_dataset, batch_size=20, shuffle = True, collate_fn = collate_val)
load_path_model = '/cluster/scratch/arsood/qwen_with_fut_traj_wo_past_traj'
model = QwenFineTunedModelText(cache_model = load_path_model)

# torch.manual_seed(42)

# training_covla_annotated = CovlaDatasetTraining(data_covla.video, data_covla.state, 4)
# training_waymo_annotated = WaymoE2EDatasetTraining(data_waymo_train.data, 4)
# training_data = ConcatDataset([training_covla_annotated])
# val_loader = DataLoader(training_data, batch_size=1, shuffle = True, collate_fn = train_collate)

# dataset_size = len(training_data)
# val_size = int(0.1 * dataset_size)
# train_size = dataset_size - val_size

# train_dataset, val_dataset = random_split(training_data, [train_size, val_size])

# save_val_dest_path = '/cluster/scratch/arsood/qwen_with_fut_traj_wo_past_traj'
# save_final_dest_path = '/cluster/scratch/arsood/qwen_with_fut_traj_wo_past_traj'
# tensor_board_path = '/cluster/scratch/arsood/tensor_board_fut_traj_wo_past_traj'
# model = QwenFineTunedModelText()
# trainer(model, training_data, 2, 2, train_collate, val_dataset=val_dataset, val_steps= 100, save_val_dest_path = save_val_dest_path, save_final_dest_path = save_final_dest_path, tensor_board_path = tensor_board_path)



# model_name = "Qwen/Qwen2.5-VL-3B-Instruct"  # Replace with the exact model name if different
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, cache_dir = "/cluster/scratch/arsood/cache_hugging_face", torch_dtype="auto",
#     device_map="auto")
# tokenizer = AutoProcessor.from_pretrained(model_name, cache_dir ="/cluster/scratch/arsood/cache_hugging_face")

# l = 0
# sum_3 = 0
# sum_5 = 0
list_3 = []
list_5 = []
l_every = []
conta = 0
# path_to_save = "/cluster/scratch/arsood/save_val_result"
for batch in tqdm(val_loader):
  # if np.any(np.abs(batch["next_state_traj"][0][...,1]) > 3):
  #   plt.figure()
  #   plt.xlim(-30, 30)
  #   plt.plot(batch["next_state_traj"][0][...,1], batch["next_state_traj"][0][...,0], color="red")
  #   plt.savefig("/cluster/home/arsood/Semester_Project_Official/fig_2.png")
  #   input_l = input("lol")
    # conta+=1
    # print(conta)

    loss_avg, pred_traj, output_text = model.validate_traj_generate(batch)
    # # print(batch["front_images"][0].shape)
    # # print(output_text[0])
    # for i in range(0, len(batch["front3_camera_image_list"])):
    #   front3_camera_image_list = batch["front3_camera_image_list"][i]
    #   front3_camera_calibration_list = batch["front_calib_matrix"][i]
    #   vehicle_pose = batch["vehicle_pose"][i]
    #   pred_traj_temp = pred_traj[i]
    #   next_state_traj_pred = np.stack([pred_traj_temp[...,0], pred_traj_temp[...,1], np.zeros_like(pred_traj_temp[...,0])], axis=1)
    #   next_state_gt = batch["next_state_traj"][i]
    #   next_state_gt = np.stack([next_state_gt[...,0], next_state_gt[...,1], np.zeros_like(next_state_gt[...,0])], axis=1)
    #   for j in range(len(front3_camera_image_list)):
    #     waypoints_camera_space = project_vehicle_to_image(vehicle_pose, front3_camera_calibration_list[j], next_state_traj_pred)
    #     waypoints_camera_space_second = project_vehicle_to_image(vehicle_pose, front3_camera_calibration_list[j], next_state_gt)
    #     front3_camera_image_list[j] = draw_points_on_image(front3_camera_image_list[j], waypoints_camera_space, size=15, colour=(0,255,0))
    #     front3_camera_image_list[j] = draw_points_on_image(front3_camera_image_list[j], waypoints_camera_space_second, size=15, colour=(0,0,255))
    #   front_concatenated = np.concatenate(front3_camera_image_list, axis=1)
    #   list_add = [front_concatenated, output_text[i], batch["index_intent"][i], next_state_gt, next_state_traj_pred]
    #   l_every.append(list_add)
    #   plt.figure()
    #   plt.imshow(front_concatenated)
    #   plt.savefig("/cluster/home/arsood/Semester_Project_Official/all_fig.png")
    ade_3_second = torch.mean(loss_avg[...,0:12].clone()).item()
    ade_5_second = torch.mean(loss_avg).item()
    list_3.append(ade_3_second)
    list_5.append(ade_5_second)
    print("Ade3: ", statistics.mean(list_3))
    print("Ade5: ", statistics.mean(list_5))
#np.save("/cluster/scratch/arsood/save_val_result/saveall_2.npy", l_every, allow_pickle=True)
