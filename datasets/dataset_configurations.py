import os
import pathlib


# Configurations
# DATA_CROP_STRIDE = 16
# DATA_CROP_SIZE = 32

# TODO: TEST: 48
DATA_CROP_STRIDE = 24
DATA_CROP_SIZE = 48

# Setups
DATA_3D_STRIDE = (DATA_CROP_STRIDE, DATA_CROP_STRIDE, DATA_CROP_STRIDE)
DATA_3D_SIZE = (DATA_CROP_SIZE, DATA_CROP_SIZE, DATA_CROP_SIZE)
DATA_2D_SIZE = (DATA_CROP_SIZE, DATA_CROP_SIZE)

# Define dataset folder
DATASET_FOLDER = "Pipes3DGeneratorCycles"  # TODO: Read from config file

# Define paths
ROOT_PATH = pathlib.Path(__file__).resolve().parent.parent
DATA_PATH = os.path.join(ROOT_PATH, "data")
DATASET_PATH = os.path.join(DATA_PATH, DATASET_FOLDER)
TRAIN_CROPPED_PATH = os.path.join(DATA_PATH, "train_cropped_data", DATASET_FOLDER)
EVAL_CROPPED_PATH = os.path.join(DATA_PATH, "eval_cropped_data", DATASET_FOLDER)
RESULTS_PATH = os.path.join(ROOT_PATH, "data_results", DATASET_FOLDER)

MODEL_RESULTS_PATH = os.path.join(RESULTS_PATH, "models")
PREDICT_PIPELINE_RESULTS_PATH = os.path.join(RESULTS_PATH, "predict_pipeline")
MERGE_PIPELINE_RESULTS_PATH = os.path.join(RESULTS_PATH, "merge_pipeline")
VISUALIZATION_RESULTS_PATH = os.path.join(RESULTS_PATH, "visualization")

# Dataset 2D
APPLY_CONTINUITY_FIX = True
V1_2D_DATASETS = ['Trees2DV1', 'Trees2DV1S']
V2_2D_DATASETS = ['Trees2DV2', 'Trees2DV2M']

# Dataset 3D
V1_3D_DATASETS = ['Trees3DV1']
V2_3D_DATASETS = ['Trees3DV2', 'Trees3DV2M', 'Trees3DV3', 'Trees3DV4']

IMAGES_6_VIEWS = ["top", "bottom", "front", "back", "left", "right"]
PROJECTION_MODE = "visualization"  # "visualization" or "training"

# Logs
TRAIN_LOG_PATH = os.path.join(TRAIN_CROPPED_PATH, "log.csv")
EVAL_LOG_PATH = os.path.join(EVAL_CROPPED_PATH, "log.csv")
