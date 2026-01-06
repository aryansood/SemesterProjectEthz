from dataclasses import dataclass

@dataclass
class DataCovla:
    video: str = ""
    state: str = ""
    annotation: str = ""
    
@dataclass
class DataTrainWaymo:
    data: str = ""
    annotation: str = ""

@dataclass
class DataValWaymo:
    data: str = ""
    annotation: str = ""

@dataclass
class DataTestWaymo:
    data: str = ""

data_covla = DataCovla(
    video="/cluster/scratch/arsood/covla/videos",
    state="/cluster/scratch/arsood/covla/states",
    annotation="/cluster/scratch/arsood/covla_annot"
)

data_waymo_train = DataTrainWaymo(
    data="/cluster/scratch/arsood/data_clean",
    annotation="/cluster/scratch/arsood/waymo_annot"
)

data_waymo_val = DataValWaymo(
    data="/cluster/scratch/arsood/data_clean_val",
)

data_waymo_test = DataTestWaymo(
    data="/cluster/scratch/arsood/test_clean"
)