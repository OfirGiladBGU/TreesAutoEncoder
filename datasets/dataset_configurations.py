import os
import pathlib
import yaml
from enum import Enum


# TODO: Setup Configurations File
CONFIG_FILENAME = "PipeForge3DPCD.yaml"

#####################
# Automatic Parsing #
#####################

class TaskType(Enum):
    SINGLE_COMPONENT = 1  # Assumption: Need to achieve a single component
    LOCAL_CONNECTIVITY = 2  # Assumption: Need to connect components on focused scope
    PATCH_HOLES = 3  # Assumption: Need to fix any type of holes
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

# Data Configurations
DATA_CROP_STRIDE = config_data.get("DATA_CROP_STRIDE", 16)
DATA_CROP_SIZE = config_data.get("DATA_CROP_SIZE", 32)
TASK_TYPE = task_type_map.get(config_data.get("TASK_TYPE", "PATCH_HOLES"))


# # TODO: TEST: 32 - Parse2022 / PipesForge3D - Mesh
# DATA_CROP_STRIDE = 16  # TODO: Read from config file
# DATA_CROP_SIZE = 32  # TODO: Read from config file

# TODO: TEST: 48 - Parse2022 / PipesForge3D - Mesh
# DATA_CROP_STRIDE = 24  # TODO: Read from config file
# DATA_CROP_SIZE = 48  # TODO: Read from config file

# # TODO: TEST: 96 - PipesForge3D - PCD
# DATA_CROP_STRIDE = 48  # TODO: Read from config file
# DATA_CROP_SIZE = 96  # TODO: Read from config file

# Setups
DATA_3D_STRIDE = (DATA_CROP_STRIDE, DATA_CROP_STRIDE, DATA_CROP_STRIDE)
DATA_3D_SIZE = (DATA_CROP_SIZE, DATA_CROP_SIZE, DATA_CROP_SIZE)
DATA_2D_SIZE = (DATA_CROP_SIZE, DATA_CROP_SIZE)

LOWER_THRESHOLD_2D = config_data.get("LOWER_THRESHOLD_2D", 0.1)
UPPER_THRESHOLD_2D = config_data.get("UPPER_THRESHOLD_2D", 0.9)

# Define dataset folder
DATASET_FOLDER = config_data.get("DATASET_FOLDER")  # TODO: Read from config file
# DATASET_FOLDER = "parse2022_48"  # TODO: Read from config file

# Data Folder Paths
DATA_PATH = os.path.join(ROOT_PATH, "data")
DATASET_PATH = os.path.join(DATA_PATH, DATASET_FOLDER)
CROPS_PATH = os.path.join(ROOT_PATH, "data_crops", DATASET_FOLDER)

# Data Results Folder Paths
RESULTS_PATH = os.path.join(ROOT_PATH, "data_results", DATASET_FOLDER)
MODEL_RESULTS_PATH = os.path.join(RESULTS_PATH, "models")
PREDICT_PIPELINE_RESULTS_PATH = os.path.join(RESULTS_PATH, "predict_pipeline")
MERGE_PIPELINE_RESULTS_PATH = os.path.join(RESULTS_PATH, "merge_pipeline")
VISUALIZATION_RESULTS_PATH = os.path.join(RESULTS_PATH, "visualization")

# Dataset 1D
V1_1D_DATASETS = ['Trees1DV1']

# Dataset 2D
APPLY_LOG_FILTER = True
# APPLY_CONTINUITY_FIX = True
APPLY_MEDIAN_FILTER = False  # For PCDs
V1_2D_DATASETS = ['Trees2DV1', 'Trees2DV1S']
V2_2D_DATASETS = ['Trees2DV2', 'Trees2DV2M']

# Dataset 3D
V1_3D_DATASETS = ['Trees3DV1']
V2_3D_DATASETS = ['Trees3DV2', 'Trees3DV2M', 'Trees3DV3', 'Trees3DV4']

# Data - Paths
LABELS = os.path.join(DATASET_PATH, "labels")  # TARGET

PREDS = os.path.join(DATASET_PATH, "preds")  # INPUT
PREDS_COMPONENTS = os.path.join(DATASET_PATH, "preds_components")

PREDS_FIXED = os.path.join(DATASET_PATH, "preds_fixed")  # INPUT (Outliers removed)
PREDS_FIXED_COMPONENTS = os.path.join(DATASET_PATH, "preds_fixed_components")

EVALS = os.path.join(DATASET_PATH, "evals")
EVALS_COMPONENTS = os.path.join(DATASET_PATH, "evals_components")


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


# Views Configurations
IMAGES_6_VIEWS = ["top", "bottom", "front", "back", "left", "right"]
PROJECTION_MODE = "visualization"  # "visualization" or "training"

# Logs
TRAIN_LOG_PATH = os.path.join(CROPS_PATH, "train_log.csv")
EVAL_LOG_PATH = os.path.join(CROPS_PATH, "eval_log.csv")
