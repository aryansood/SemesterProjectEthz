#!/usr/bin/env python3
import tensorflow as tf
import os
import numpy as np
from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2
import time

DATASET_FOLDER = '/cluster/scratch/arsood/data' #Put the location of where the trfrecords are stored

TRAIN_FILES = os.path.join(DATASET_FOLDER, 'training_*.tfrecord*')
VALIDATION_FILES = os.path.join(DATASET_FOLDER, 'val*.tfrecord*')
TEST_FILES = os.path.join(DATASET_FOLDER, 'test_*.tfrecord*')

SAVE_LOCATION = "/cluster/scratch/arsood/data_strings_train" #Put the location of where to save the processed data


name_folders_train = np.load('training_segments.npy')
name_folders_val = np.load('val_segments.npy')

name_folders = name_folders_train #Choose if using name_folders_train or name_folders_val
filenames = tf.io.matching_files(VALIDATION_FILES) #Choose beetween TRAIN_FILES, VALIDATION_FILES, TEST_FILES


for el in name_folders:
    os.makedirs(os.path.join(SAVE_LOCATION, el), exist_ok=True)

# dataset = tf.data.TFRecordDataset(filenames, compression_type='')
# dataset_iter = dataset.as_numpy_iterator()

# count = 0
# start_time = time.time()
# for bytes_example in dataset_iter:
#     data = wod_e2ed_pb2.E2EDFrame()
#     data.ParseFromString(bytes_example)
#     sequence_name, sample_idx = data.frame.context.name.split('-')
#     out_path = os.path.join(os.path.join(SAVE_LOCATION, sequence_name), f"{data.frame.context.name}.bin")
#     with open(out_path, 'wb') as f:
#         f.write(bytes_example)
#     count = count+1
#     print(count)
    
# end_time = time.time()
# print(f"Total time taken: {end_time - start_time:.2f} seconds")


