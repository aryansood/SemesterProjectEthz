from dataloader_utils.dataloader_waymo import WaymoE2EDatasetTrainingAnnotated
from torch.utils.data import DataLoader
from config import data_waymo
from tqdm import tqdm



training_dataset = WaymoE2EDatasetTrainingAnnotated(data_waymo.waymo_data, data_waymo.waymo_annotation, 4)
training_loader = DataLoader(training_dataset, 1, shuffle=False)

for batch in tqdm(training_loader):
    inputl = input("Continue?")
    l = 10