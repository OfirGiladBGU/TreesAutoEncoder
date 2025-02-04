import os
import pathlib


# Configurations
# DATA_CROP_STRIDE = 16
# DATA_CROP_SIZE = 32

# TODO: TEST: 48
DATA_CROP_STRIDE = 24  # TODO: Read from config file
DATA_CROP_SIZE = 48  # TODO: Read from config file

# Setups
DATA_3D_STRIDE = (DATA_CROP_STRIDE, DATA_CROP_STRIDE, DATA_CROP_STRIDE)
DATA_3D_SIZE = (DATA_CROP_SIZE, DATA_CROP_SIZE, DATA_CROP_SIZE)
DATA_2D_SIZE = (DATA_CROP_SIZE, DATA_CROP_SIZE)

# Define dataset folder
DATASET_FOLDER = "parse2022_48"  # TODO: Read from config file

# Data Folder Paths
ROOT_PATH = pathlib.Path(__file__).resolve().parent.parent
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
APPLY_CONTINUITY_FIX = True
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
