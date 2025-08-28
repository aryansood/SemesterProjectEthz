from dataclasses import dataclass

@dataclass
class DataCovla:
    covla_video: str = ""
    covla_state: str = ""
    covla_annotation: str = ""
    
@dataclass
class DataTrainWaymo:
    waymo_data: str = ""
    waymo_annotation: str = ""

@dataclass
class DataValWaymo:
    waymo_data: str = ""
    waymo_annotation: str = ""

data_covla = DataCovla(
    covla_video="/cluster/scratch/arsood/covla/videos",
    covla_state="/cluster/scratch/arsood/covla/states",
    covla_annotation="/cluster/scratch/arsood/covla_annot"
)

data_waymo_train = DataTrainWaymo(
    waymo_data="/cluster/scratch/arsood/data_clean",
    waymo_annotation="/cluster/scratch/arsood/waymo_annot"
)

data_waymo_val = DataValWaymo(
    waymo_data="/cluster/scratch/arsood/data_val_clean",
)