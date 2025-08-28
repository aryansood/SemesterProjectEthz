from dataloader_utils.dataloader_waymo import WaymoE2EDatasetTrainingAnnotated
from dataloader_utils.dataloader_covla import CovlaDatasetTrainingAnnotated
from torch.utils.data import DataLoader
from config import data_waymo_train, data_covla
from tqdm import tqdm



training_dataset = CovlaDatasetTrainingAnnotated(data_covla.covla_video, data_covla.covla_state, data_covla.covla_annotation, 4)
training_loader = DataLoader(training_dataset, 1, shuffle=False)

for batch in tqdm(training_loader):
    l = 10

