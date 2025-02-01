import os
import pathlib


# Configurations
APPLY_LOG_FILTER = True

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
DATASET_FOLDER = "Pipes3DGeneratorCycles"  # TODO: Read from config file

# Data Folder Paths
ROOT_PATH = pathlib.Path(__file__).resolve().parent.parent
DATA_PATH = os.path.join(ROOT_PATH, "data")
DATASET_PATH = os.path.join(DATA_PATH, DATASET_FOLDER)
TRAIN_CROPPED_PATH = os.path.join(DATA_PATH, "train_cropped_data", DATASET_FOLDER)
EVAL_CROPPED_PATH = os.path.join(DATA_PATH, "eval_cropped_data", DATASET_FOLDER)

# Data Results Folder Paths
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

# Data - Paths
LABELS = os.path.join(DATASET_PATH, "labels")
PREDS = os.path.join(DATASET_PATH, "preds")
PREDS_COMPONENTS = os.path.join(DATASET_PATH, "preds_components")
EVALS = os.path.join(DATASET_PATH, "evals")
EVALS_COMPONENTS = os.path.join(DATASET_PATH, "evals_components")

# Data 2D - Paths
LABELS_2D = os.path.join(TRAIN_CROPPED_PATH, "labels_2d_v6")
PREDS_2D = os.path.join(TRAIN_CROPPED_PATH, "preds_2d_v6")
PREDS_FIXED_2D = os.path.join(TRAIN_CROPPED_PATH, "preds_fixed_2d_v6")
PREDS_COMPONENTS_2D = os.path.join(TRAIN_CROPPED_PATH, "preds_components_2d_v6")
PREDS_FIXED_COMPONENTS_2D = os.path.join(TRAIN_CROPPED_PATH, "preds_fixed_components_2d_v6")

EVALS_2D = os.path.join(EVAL_CROPPED_PATH, "evals_2d_v6")
EVALS_COMPONENTS_2D = os.path.join(EVAL_CROPPED_PATH, "evals_components_2d_v6")

# Data 3D - Paths
LABELS_3D = os.path.join(TRAIN_CROPPED_PATH, "labels_3d_v6")
PREDS_3D = os.path.join(TRAIN_CROPPED_PATH, "preds_3d_v6")
PREDS_FIXED_3D = os.path.join(TRAIN_CROPPED_PATH, "preds_fixed_3d_v6")
PREDS_COMPONENTS_3D = os.path.join(TRAIN_CROPPED_PATH, "preds_components_3d_v6")
PREDS_FIXED_COMPONENTS_3D = os.path.join(TRAIN_CROPPED_PATH, "preds_fixed_components_3d_v6")

EVALS_3D = os.path.join(EVAL_CROPPED_PATH, "evals_3d_v6")
EVALS_COMPONENTS_3D = os.path.join(EVAL_CROPPED_PATH, "evals_components_3d_v6")

LABELS_3D_RECONSTRUCT = os.path.join(TRAIN_CROPPED_PATH, "labels_3d_reconstruct_v6")
PREDS_3D_RECONSTRUCT = os.path.join(TRAIN_CROPPED_PATH, "preds_3d_reconstruct_v6")
PREDS_FIXED_3D_RECONSTRUCT = os.path.join(TRAIN_CROPPED_PATH, "preds_fixed_3d_reconstruct_v6")

PREDS_3D_FUSION = os.path.join(TRAIN_CROPPED_PATH, "preds_3d_fusion_v6")
PREDS_FIXED_3D_FUSION = os.path.join(TRAIN_CROPPED_PATH, "preds_fixed_3d_fusion_v6")

# Views Configurations
IMAGES_6_VIEWS = ["top", "bottom", "front", "back", "left", "right"]
PROJECTION_MODE = "visualization"  # "visualization" or "training"

# Logs
TRAIN_LOG_PATH = os.path.join(TRAIN_CROPPED_PATH, "log.csv")
EVAL_LOG_PATH = os.path.join(EVAL_CROPPED_PATH, "log.csv")
