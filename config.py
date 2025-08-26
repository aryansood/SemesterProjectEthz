from dataclasses import dataclass

@dataclass
class DataCovla:
    covla_video: str = ""
    covla_state: str = ""
    covla_annotation: str = ""
    
@dataclass
class DataWaymo:
    waymo_data: str = ""
    waymo_annotation: str = ""



data_covla = DataCovla(
    covla_video="/cluster/scratch/arsood/covla/videos",
    covla_state="/cluster/scratch/arsood/covla/states",
    covla_annotation="/cluster/scratch/arsood/covla_string"
)

data_waymo = DataWaymo(
    waymo_data="/cluster/scratch/arsood/data_clean",
    waymo_annotation="/cluster/scratch/arsood/data_strings_train"
)