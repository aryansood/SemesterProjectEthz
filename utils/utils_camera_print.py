import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
import matplotlib.pyplot as plt
import statistics
from tqdm import tqdm
import tensorflow as tf
from waymo_open_dataset.wdl_limited.camera.ops import py_camera_model_ops
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2
import cv2
import os
import numpy as np


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


def validate_proj_save(val_loader, model, path_to_save):
    os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
    list_to_save_all = []
    for batch in tqdm(val_loader):
        loss_avg, pred_traj = model.generate_traj_val(batch)
        for i in range(0, len(batch["front3_camera_image_list"])):
            front3_camera_image_list = batch["front3_camera_image_list"][i]
            front3_camera_calibration_list = batch["front_calib_matrix"][i]
            vehicle_pose = batch["vehicle_pose"][i]
            pred_traj_temp = pred_traj[i]
            next_state_traj_pred = np.stack([pred_traj_temp[...,0], pred_traj_temp[...,1], np.zeros_like(pred_traj_temp[...,0])], axis=1)
            next_state_gt = batch["next_state_traj"][i]
            next_state_gt = np.stack([next_state_gt[...,0], next_state_gt[...,1], np.zeros_like(next_state_gt[...,0])], axis=1)
            for j in range(len(front3_camera_image_list)):
                waypoints_camera_space = project_vehicle_to_image(vehicle_pose, front3_camera_calibration_list[j], next_state_traj_pred)
                waypoints_camera_space_second = project_vehicle_to_image(vehicle_pose, front3_camera_calibration_list[j], next_state_gt)
                front3_camera_image_list[j] = draw_points_on_image(front3_camera_image_list[j], waypoints_camera_space, size=15, colour=(0,255,0))
                front3_camera_image_list[j] = draw_points_on_image(front3_camera_image_list[j], waypoints_camera_space_second, size=15, colour=(0,0,255))
            front_concatenated = np.concatenate(front3_camera_image_list, axis=1)
            list_add = [front_concatenated, batch["index_intent"][i], next_state_gt, next_state_traj_pred]
            list_to_save_all.append(list_add)
        
    np.save(path_to_save, np.array(list_to_save_all, dtype=object), allow_pickle=True)