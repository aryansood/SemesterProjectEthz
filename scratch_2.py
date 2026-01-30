from dataloader_utils_2.waymo_test_dataloader import WaymoE2EDatasetTest, test_collate
from torch.utils.data import DataLoader
from config import data_waymo_test
from tqdm import tqdm
from model.qwen_cot_latent import QwenLatentCot
from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2
from waymo_open_dataset.protos import end_to_end_driving_submission_pb2 as wod_e2ed_submission_pb2
import os
import tensorflow as tf
import math

waymo_test_dataset = WaymoE2EDatasetTest(data_waymo_test.data, 4)
waymo_dataloader = DataLoader(waymo_test_dataset, batch_size=20, shuffle=False, collate_fn=test_collate)

# path_val_virt = "/cluster/scratch/arsood/cot_latent_val_spline_7/cot_latent_val_virt.pt"
# path_val_linear = "/cluster/scratch/arsood/cot_latent_val_spline_7/cot_latent_val_linear.pt"

# path_val_virt = "/cluster/scratch/arsood/cot_latent_val_spline/cot_latent_val_virt.pt"
# path_val_linear = "/cluster/scratch/arsood/cot_latent_val_spline/cot_latent_val_linear.pt"

path_val_virt = "/cluster/scratch/arsood/cot_latent_val_spline_9/cot_latent_val_virt.pt"
path_val_linear = "/cluster/scratch/arsood/cot_latent_val_spline_9/cot_latent_val_linear.pt"

prcoessor_config = "/cluster/scratch/arsood/Qwen_2_5_vlm"
model_to_load_2 = "/cluster/scratch/arsood/Qwen_Fine_Tune_Waymo_Data_Final"
model = QwenLatentCot(model_to_load_2, prcoessor_config)
model.load_model(path_val_virt, path_val_linear)

predictions = []

for batch in tqdm(waymo_dataloader):
    pred = model.generate_trajectory(batch)
    for index_num, el in enumerate(batch["frame_names"]):
        traj_to_pred = pred[index_num]
        predicted_trajectory = wod_e2ed_submission_pb2.TrajectoryPrediction(pos_x=traj_to_pred[...,0], pos_y=traj_to_pred[...,1])
        frame_trajectory = wod_e2ed_submission_pb2.FrameTrajectoryPredictions(frame_name=el, trajectory=predicted_trajectory)
        predictions.append(frame_trajectory)


num_submission_shards = len(predictions)
submission_file_base = '/cluster/scratch/arsood/submit_9'
if not os.path.exists(submission_file_base):
  os.makedirs(submission_file_base)
sub_file_names = [
    os.path.join(submission_file_base, part)
    for part in [f'mysubmission.binproto-{i:05d}-of-{num_submission_shards:05d}' for i in range(num_submission_shards)]
]

submissions = []
num_predictions_per_shard =  math.ceil(len(predictions) / num_submission_shards)
for i in range(num_submission_shards):
  start = i * num_predictions_per_shard
  end = (i + 1) * num_predictions_per_shard
  submissions.append(
      wod_e2ed_submission_pb2.E2EDChallengeSubmission(
          predictions=predictions[start:end]))
  
for i, shard in enumerate(submissions):
  shard.submission_type  =  wod_e2ed_submission_pb2.E2EDChallengeSubmission.SubmissionType.E2ED_SUBMISSION
  shard.authors[:] = ['Aryan', 'Sood']  # Please modify accordingly.
  shard.affiliation = 'ETHZurich'  # Please modify accordingly.
  shard.account_name = 'soodaryan972@gmail.com'  # Please modify accordingly.
  shard.unique_method_name = 'VLM-GRPO-FULL-TRAJ'  # Please modify accordingly.
  shard.method_link = 'method_link'  # Please modify accordingly.
  shard.description = ''  # Please modify accordingly.
  shard.uses_public_model_pretraining = True # Please modify accordingly.
  shard.public_model_names.extend(['Qwen']) # Please modify accordingly.
  shard.num_model_parameters = "200k" # Please modify accordingly.
  with tf.io.gfile.GFile(sub_file_names[i], 'wb') as fp:
    fp.write(shard.SerializeToString())
print(submissions[0])
     
