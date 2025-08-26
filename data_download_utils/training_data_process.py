#!/usr/bin/env python3
import tensorflow as tf
import os
import numpy as np
from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2
import time
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--dir", type=str, required=True, help="Input directory")
parser.add_argument("--save", type=str, required=True, help="Output directory")
parser.add_argument("--type", type=str, help="Choose val or train")

args = parser.parse_args()


DATASET_FOLDER = args.dir #'/cluster/scratch/arsood/data'

TRAIN_FILES = os.path.join(DATASET_FOLDER, 'training_*.tfrecord*')
VALIDATION_FILES = os.path.join(DATASET_FOLDER, 'val*.tfrecord*')
TEST_FILES = os.path.join(DATASET_FOLDER, 'test_*.tfrecord*')

SAVE_LOCATION = args.save #"/cluster/scratch/arsood/data_strings_train"


name_folders_train = np.load('training_segments.npy')
name_folders_val = np.load('val_segments.npy')

FILES_TYPE = None
name_folders = None

if(args.type == 'val'):
    FILES_TYPE = VALIDATION_FILES
    name_folders = name_folders_val

elif(args.type == 'train'):
    FILES_TYPE = TRAIN_FILES
    name_folders = name_folders_train

filenames = tf.io.matching_files(FILES_TYPE)


for el in name_folders:
    os.makedirs(os.path.join(SAVE_LOCATION, el), exist_ok=True)

dataset = tf.data.TFRecordDataset(filenames, compression_type='')
dataset_iter = dataset.as_numpy_iterator()

count = 0
start_time = time.time()
for bytes_example in dataset_iter:
    data = wod_e2ed_pb2.E2EDFrame()
    data.ParseFromString(bytes_example)
    sequence_name, sample_idx = data.frame.context.name.split('-')
    out_path = os.path.join(os.path.join(SAVE_LOCATION, sequence_name), f"{data.frame.context.name}.bin")
    with open(out_path, 'wb') as f:
        f.write(bytes_example)
    count = count+1
    print(count)
    
end_time = time.time()
print(f"Total time taken: {end_time - start_time:.2f} seconds")


