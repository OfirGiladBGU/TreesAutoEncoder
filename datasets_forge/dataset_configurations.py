import os
import pathlib
import yaml
import math
from enum import Enum


# TODO: Setup Configurations File
CONFIG_FILENAME = "parse2022_SC_32.yaml"


# TODO: Use for examples:

# CONFIG_FILENAME = "examples/parse2022_LC_32.yaml"
# CONFIG_FILENAME = "examples/parse2022_LC_48.yaml"

# CONFIG_FILENAME = "examples/PipeForge3DMesh_LC_32.yaml"
# CONFIG_FILENAME = "examples/PipeForge3DMesh_LC_48.yaml"

# CONFIG_FILENAME = "examples/PipeForge3DPCD_LC_32.yaml"
# CONFIG_FILENAME = "examples/PipeForge3DPCD_LC_48.yaml"

# CONFIG_FILENAME = "examples/PipeForge3DTree_32.yaml"

# TODO: Use for Ex1:

# CONFIG_FILENAME = "experiment1/parse2022_LC_32_50.yaml"
# CONFIG_FILENAME = "experiment1/PipeForge3DMesh_Best_LC_32.yaml"
# CONFIG_FILENAME = "experiment1/PipeForge3DPCD_Best_LC_32.yaml"
# CONFIG_FILENAME = "experiment1/HospitalPCD_LC_32.yaml"


# TODO: Use for Ex2:

# V1 - 20 samples

# CONFIG_FILENAME = "experiment2/parse2022_LC_32.yaml"
# CONFIG_FILENAME = "experiment2/parse2022_LC_48.yaml"
# CONFIG_FILENAME = "experiment2/parse2022_LC_64.yaml"
# CONFIG_FILENAME = "experiment2/parse2022_LC_96.yaml"


# V2 - 50 samples

# CONFIG_FILENAME = "experiment2/parse2022_LC_32_50.yaml"
# CONFIG_FILENAME = "experiment2/parse2022_LC_48_50.yaml"
# CONFIG_FILENAME = "experiment2/parse2022_LC_64_50.yaml"
# CONFIG_FILENAME = "experiment2/parse2022_LC_96_50.yaml"


# TODO: Use for Ex3:
# REDO with trained PipeForge3DMesh_LC_32_Base - size 1.0 , scale 0.25

# CONFIG_FILENAME = "experiment3/PipeForge3DMesh_Base_LC_32.yaml"

# CONFIG_FILENAME = "experiment3/PipeForge3DMesh_150_LC_32.yaml"
# CONFIG_FILENAME = "experiment3/PipeForge3DMesh_200_LC_32.yaml"
# CONFIG_FILENAME = "experiment3/PipeForge3DMesh_300_LC_32.yaml"
# CONFIG_FILENAME = "experiment3/PipeForge3DMesh_400_LC_32.yaml"


# TODO: Use for Parse2022 test:
# CONFIG_FILENAME = "parse2022_test/parse2022_LC_32_test.yaml"


# TODO: Use for hyperparams tests:

# CONFIG_FILENAME = "tests/parse2022_LC_32_v1.yaml"
# CONFIG_FILENAME = "tests/parse2022_LC_32_v2.yaml"
# CONFIG_FILENAME = "tests/parse2022_LC_32_v3.yaml"
# CONFIG_FILENAME = "tests/parse2022_LC_32_v4.yaml"
# CONFIG_FILENAME = "tests/parse2022_LC_32_v5.yaml"


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

class ProjectionMode(Enum):  # TODO: Not yet used
    VISUAL_MODE = 1  # Same as looking on a Rubiks from 6 view direction (With rotation matrices)
    MATHEMATICAL_MODE = 2  # Direct projection on each of the 6 planes

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

# Preparation
APPLY_LOG_FILTER = True  # Helpful to reject projections with too dense or too sparse pixels from the training loader
APPLY_MEDIAN_FILTER = False  # Sometimes helpful for PCDs to fill missing inner black pixels
APPLY_CONTINUITY_FIX_3D = True  # The continuity filter for true holes detection in 3D
APPLY_CONTINUITY_FIX_2D = False  # The continuity filter for true holes detection in 2D
BINARY_DILATION = True  # Enable the usage local scope binary dilation mask (2D - 4 directions, 3D - 6 directions)
CONNECTIVITY_TYPE_3D = 6
CONNECTIVITY_TYPE_2D = 4

# Dataset 1D
V1_1D_DATASETS = ['Trees1DV1']

# Dataset 2D
V1_2D_DATASETS = ['Trees2DV1', 'Trees2DV1S']
V2_2D_DATASETS = ['Trees2DV2', 'Trees2DV2M']

# Dataset 3D
V1_3D_DATASETS = ['Trees3DV1']
V2_3D_DATASETS = ['Trees3DV2', 'Trees3DV2M', 'Trees3DV3', 'Trees3DV4']

# Views Configurations
IMAGES_6_VIEWS = ["top", "bottom", "front", "back", "left", "right"]
PROJECTION_MODE = ProjectionMode.VISUAL_MODE

###################
# DATASET - PATHS #
###################

LABELS = os.path.join(DATASET_PATH, "labels")  # TARGET
LABELS_COMPONENTS = os.path.join(DATASET_PATH, "labels_components")

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
LABELS_COMPONENTS_2D = os.path.join(CROPS_PATH, "labels_components_2d")

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
LABELS_COMPONENTS_3D = os.path.join(CROPS_PATH, "labels_components_3d")

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
