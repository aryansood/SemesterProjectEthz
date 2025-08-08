## Downloading the Waymo Open Dataset (End-to-End Camera Challenge)

### Prerequisites

1. Install the [Google Cloud SDK](https://cloud.google.com/sdk/docs/install), which includes `gsutil`.
2. Authenticate with Google Cloud:
   ```bash
   gcloud auth login
   ```
3. Accept access permissions on the [Waymo Open Dataset Challenge site](https://waymo.com/open/download/).

---

### Step 1: Create Folders for the Dataset

```bash
mkdir -p waymo_data/training
mkdir -p waymo_data/validation

```

---

### Step 2: Download the Training Data

```bash
cd waymo_data/training
gsutil -m cp -r gs://waymo_open_dataset_end_to_end_camera_v_1_0_0/training_*.tfrecord* training/
```

---

### Step 3: Download the Validation Data

```bash
cd waymo_data/validation
gsutil -m cp -r gs://waymo_open_dataset_end_to_end_camera_v_1_0_0/val*.tfrecord* validation/
```
---

### Step 4: Process the dataset
Open the (training_data_process.py) and change the following line, depending if you want to 
```python
DATASET_FOLDER = 'waymo_data/training' #Put the location of where the trfrecords are stored
...

SAVE_LOCATION = "/cluster/scratch/arsood/data_new" #Put the location of where to save the processed data
...
name_folders = name_folders_train #Choose if using name_folders_train or name_folders_val
filenames = tf.io.matching_files(VALIDATION_FILES) #Choose beetween TRAIN_FILES, VALIDATION_FILES, TEST_FILES
```
then do:
```bash
python training_data_process.py
```



