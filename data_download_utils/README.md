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
gsutil -m cp -r gs://waymo_open_dataset_end_to_end_camera_v_1_0_0/training_*.tfrecord* .
```

---

### Step 3: Download the Validation Data

```bash
cd waymo_data/validation
gsutil -m cp -r gs://waymo_open_dataset_end_to_end_camera_v_1_0_0/val*.tfrecord* .
```
---

### Step 4: Process the dataset
Now run the following code 
then do:
```bash
python training_data_process.py \
    --dir <path_to_tfrecord_files> \
    --save <path_to_save_processed_data> \
    --type <train_or_val>

```
#### Example for training data:
```bash
python training_data_process.py --dir waymo_data/training --save processed/training --type train
```

#### Example for validation data:
```bash
python training_data_process.py --dir waymo_data/validation --save processed/validation --type val
```



