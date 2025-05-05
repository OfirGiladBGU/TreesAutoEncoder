import os
import pathlib
import yaml
import math
from enum import Enum


# TODO: Setup Configurations File
CONFIG_FILENAME = "parse2022_LC_32.yaml"

#####################
# Automatic Parsing #
#####################

# Util classes
class ModelType:
    Model_1D = 1
    Model_2D = 2
    Model_3D = 3

class DataType(Enum):
    TRAIN = 1
    EVAL = 2

class TaskType(Enum):
    SINGLE_COMPONENT = 1  # Assumption: Need to achieve a single component
    LOCAL_CONNECTIVITY = 2  # Assumption: Need to connect components on focused scope
    PATCH_HOLES = 3  # Assumption: Need to fix any type of holes

# YAML text to Enum mapping
task_type_map = {
    "SINGLE_COMPONENT": TaskType.SINGLE_COMPONENT,
    "LOCAL_CONNECTIVITY": TaskType.LOCAL_CONNECTIVITY,
    "PATCH_HOLES": TaskType.PATCH_HOLES
}

# Root Path
ROOT_PATH = pathlib.Path(__file__).resolve().parent.parent

# Read configurations from config file
CONFIG_FILEPATH = os.path.join(ROOT_PATH, "configs", CONFIG_FILENAME)
with open(CONFIG_FILEPATH, 'r') as stream:
    config_data: dict = yaml.safe_load(stream)

# Read Data Configurations
DATASET_INPUT_FOLDER = config_data.get("DATASET_INPUT_FOLDER", None)
DATASET_OUTPUT_FOLDER = config_data.get("DATASET_OUTPUT_FOLDER", None)
DATA_CROP_STRIDE = config_data.get("DATA_CROP_STRIDE", 16)
DATA_CROP_SIZE = config_data.get("DATA_CROP_SIZE", 32)
LOWER_THRESHOLD_2D = config_data.get("LOWER_THRESHOLD_2D", 0.1)
UPPER_THRESHOLD_2D = config_data.get("UPPER_THRESHOLD_2D", 0.9)
TASK_TYPE = task_type_map.get(config_data.get("TASK_TYPE", "PATCH_HOLES"))
START_INDEX = config_data.get("START_INDEX", -1)
STOP_INDEX = config_data.get("STOP_INDEX", -1)

# Parse Data Configurations
if DATASET_INPUT_FOLDER is None:
    raise ValueError("DATASET_INPUT_FOLDER is not defined in the configuration file")
DATASET_OUTPUT_FOLDER = DATASET_INPUT_FOLDER if DATASET_OUTPUT_FOLDER is None else DATASET_OUTPUT_FOLDER

DATA_3D_STRIDE = (DATA_CROP_STRIDE, DATA_CROP_STRIDE, DATA_CROP_STRIDE)
DATA_3D_SIZE = (DATA_CROP_SIZE, DATA_CROP_SIZE, DATA_CROP_SIZE)
DATA_2D_SIZE = (DATA_CROP_SIZE, DATA_CROP_SIZE)
LOWER_THRESHOLD_2D *= math.pow(DATA_CROP_SIZE, 2)
UPPER_THRESHOLD_2D *= math.pow(DATA_CROP_SIZE, 2)

# Data Folder Paths
DATA_PATH = os.path.join(ROOT_PATH, "data")
DATA_CROPS_PATH = os.path.join(ROOT_PATH, "data_crops")
DATA_RESULTS_PATH = os.path.join(ROOT_PATH, "data_results")

# Dataset Folder Paths
DATASET_PATH = os.path.join(DATA_PATH, DATASET_INPUT_FOLDER)
CROPS_PATH = os.path.join(DATA_CROPS_PATH, DATASET_OUTPUT_FOLDER)
RESULTS_PATH = os.path.join(DATA_RESULTS_PATH, DATASET_OUTPUT_FOLDER)

#########
# FLAGS #
#########

# Dataset 1D
V1_1D_DATASETS = ['Trees1DV1']

# Dataset 2D
APPLY_LOG_FILTER = True
APPLY_MEDIAN_FILTER = False  # For PCDs
APPLY_CONTINUITY_FIX_2D = True
APPLY_CONTINUITY_FIX_3D = True
BINARY_DILATION = True
V1_2D_DATASETS = ['Trees2DV1', 'Trees2DV1S']
V2_2D_DATASETS = ['Trees2DV2', 'Trees2DV2M']

# Dataset 3D
V1_3D_DATASETS = ['Trees3DV1']
V2_3D_DATASETS = ['Trees3DV2', 'Trees3DV2M', 'Trees3DV3', 'Trees3DV4']

# Views Configurations
IMAGES_6_VIEWS = ["top", "bottom", "front", "back", "left", "right"]
PROJECTION_MODE = "visualization"  # "visualization" or "training"

###################
# DATASET - PATHS #
###################

LABELS = os.path.join(DATASET_PATH, "labels")  # TARGET

PREDS = os.path.join(DATASET_PATH, "preds")  # INPUT
PREDS_COMPONENTS = os.path.join(DATASET_PATH, "preds_components")

PREDS_FIXED = os.path.join(DATASET_PATH, "preds_fixed")  # INPUT (Outliers removed)
PREDS_FIXED_COMPONENTS = os.path.join(DATASET_PATH, "preds_fixed_components")

EVALS = os.path.join(DATASET_PATH, "evals")
EVALS_COMPONENTS = os.path.join(DATASET_PATH, "evals_components")

#########################
# DATASET CROPS - PATHS #
#########################

# Data 2D - Paths

# TRAINING
LABELS_2D = os.path.join(CROPS_PATH, "labels_2d")  # TARGET

PREDS_2D = os.path.join(CROPS_PATH, "preds_2d")  # INPUT
PREDS_COMPONENTS_2D = os.path.join(CROPS_PATH, "preds_components_2d")

PREDS_FIXED_2D = os.path.join(CROPS_PATH, "preds_fixed_2d")  # INPUT (Outliers removed)
PREDS_FIXED_COMPONENTS_2D = os.path.join(CROPS_PATH, "preds_fixed_components_2d")

PREDS_ADVANCED_FIXED_2D = os.path.join(CROPS_PATH, "preds_advanced_fixed_2d")  # INPUT (Continuity fixed)
PREDS_ADVANCED_FIXED_COMPONENTS_2D = os.path.join(CROPS_PATH, "preds_advanced_fixed_components_2d")

# EVALUATION
EVALS_2D = os.path.join(CROPS_PATH, "evals_2d")
EVALS_COMPONENTS_2D = os.path.join(CROPS_PATH, "evals_components_2d")


# Data 3D - Paths

# TRAINING
LABELS_3D = os.path.join(CROPS_PATH, "labels_3d")  # TARGET
# NOTE: Labels components are not used

PREDS_3D = os.path.join(CROPS_PATH, "preds_3d")  # INPUT
PREDS_COMPONENTS_3D = os.path.join(CROPS_PATH, "preds_components_3d")

PREDS_FIXED_3D = os.path.join(CROPS_PATH, "preds_fixed_3d")  # INPUT (Outliers removed)
PREDS_FIXED_COMPONENTS_3D = os.path.join(CROPS_PATH, "preds_fixed_components_3d")

PREDS_ADVANCED_FIXED_3D = os.path.join(CROPS_PATH, "preds_advanced_fixed_3d")  # INPUT (Continuity fixed)
PREDS_ADVANCED_FIXED_COMPONENTS_3D = os.path.join(CROPS_PATH, "preds_advanced_fixed_components_3d")

# RECONSTRUCT PATHS
LABELS_3D_RECONSTRUCT = os.path.join(CROPS_PATH, "labels_3d_reconstruct")  # INPUT (Direct repair)
PREDS_3D_RECONSTRUCT = os.path.join(CROPS_PATH, "preds_3d_reconstruct")  # INPUT (Direct repair)
PREDS_FIXED_3D_RECONSTRUCT = os.path.join(CROPS_PATH, "preds_fixed_3d_reconstruct")  # INPUT (Direct repair)
PREDS_ADVANCED_FIXED_3D_RECONSTRUCT = os.path.join(CROPS_PATH, "preds_advanced_fixed_3d_reconstruct")  # INPUT (Direct repair)

# FUSION PATHS
PREDS_3D_FUSION = os.path.join(CROPS_PATH, "preds_3d_fusion")  # INPUT (Fusion data)
PREDS_FIXED_3D_FUSION = os.path.join(CROPS_PATH, "preds_fixed_3d_fusion")  # INPUT (Fusion data)
PREDS_ADVANCED_FIXED_3D_FUSION = os.path.join(CROPS_PATH, "preds_advanced_fixed_3d_fusion")  # INPUT (Fusion data)

# EVALUATION
EVALS_3D = os.path.join(CROPS_PATH, "evals_3d")
EVALS_COMPONENTS_3D = os.path.join(CROPS_PATH, "evals_components_3d")

# LOGS
TRAIN_LOG_PATH = os.path.join(CROPS_PATH, "train_log.csv")
EVAL_LOG_PATH = os.path.join(CROPS_PATH, "eval_log.csv")

###########################
# DATASET RESULTS - PATHS #
###########################

MODELS_RESULTS_PATH = os.path.join(RESULTS_PATH, "models")
PREDICT_PIPELINE_RESULTS_PATH = os.path.join(RESULTS_PATH, "predict_pipeline")
MERGE_PIPELINE_RESULTS_PATH = os.path.join(RESULTS_PATH, "merge_pipeline")
VISUALIZATION_RESULTS_PATH = os.path.join(RESULTS_PATH, "visualization")

PREDICT_PIPELINE_DICE_CSV_FILES_PATH = os.path.join(PREDICT_PIPELINE_RESULTS_PATH, "csv_files")

# TODO: Add Predict Pipeline Configs
