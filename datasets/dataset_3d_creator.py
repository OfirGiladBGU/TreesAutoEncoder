import numpy as np
import os
import pathlib
import pandas as pd
from tqdm import tqdm

from dataset_list import DATA_PATH, TRAIN_CROPPED_PATH
from dataset_utils import (IMAGES_6_VIEWS,
                           get_data_file_stem, convert_data_file_to_numpy, convert_numpy_to_data_file,
                           project_3d_to_2d, get_images_6_views, reconstruct_3d_from_2d)
# TODO: Debug Tools
from dataset_visulalization import interactive_plot_2d, interactive_plot_3d


############################
# Repair 3D reconstruction #
############################
# from scipy.ndimage import binary_erosion, binary_dilation
#
#
# def refine_voxel_grid(voxel_grid, erosion_iterations=0, dilation_iterations=1):
#     refined_voxel_grid = voxel_grid
#
#     # Apply binary erosion
#     # refined_voxel_grid = binary_erosion(refined_voxel_grid, iterations=erosion_iterations)
#
#     # Apply binary dilation
#     refined_voxel_grid = binary_dilation(refined_voxel_grid, iterations=dilation_iterations)
#     refined_voxel_grid = refined_voxel_grid.astype(np.uint8)
#
#     return refined_voxel_grid
#
#
# def refine_construction(voxel_grid: np.ndarray):
#     # Extract surface mesh using Marching Cubes
#     voxel_grid_refined = refine_voxel_grid(voxel_grid)
#
#     convert_numpy_to_nii_gz(voxel_grid_refined, save_name="Test")


def create_3d_reconstructions():
    # Sources
    source_folders = {
        "labels_3d": os.path.join(TRAIN_CROPPED_PATH, "labels_3d_v6"),
        "preds_3d": os.path.join(TRAIN_CROPPED_PATH, "preds_3d_v6"),
        "preds_fixed_3d": os.path.join(TRAIN_CROPPED_PATH, "preds_fixed_3d_v6")
    }

    # Inputs
    input_folders = {
        "labels_2d": os.path.join(TRAIN_CROPPED_PATH, "labels_2d_v6"),
        "preds_2d": os.path.join(TRAIN_CROPPED_PATH, "preds_2d_v6"),
        "preds_fixed_2d": os.path.join(TRAIN_CROPPED_PATH, "preds_fixed_2d_v6")
    }

    # Outputs
    output_folders = {
        "labels_3d_reconstruct": os.path.join(TRAIN_CROPPED_PATH, "labels_3d_reconstruct_v6"),
        "preds_3d_reconstruct": os.path.join(TRAIN_CROPPED_PATH, "preds_3d_reconstruct_v6"),
        "preds_fixed_3d_reconstruct": os.path.join(TRAIN_CROPPED_PATH, "preds_fixed_3d_reconstruct_v6")
    }

    # Create Output Folders
    for output_folder in output_folders.values():
        os.makedirs(output_folder, exist_ok=True)

    # Get the filepaths
    input_filepaths = dict()
    for key, value in input_folders.items():
        input_filepaths[key] = sorted(pathlib.Path(value).rglob("*.png"))

    log_filepath = os.path.join(TRAIN_CROPPED_PATH, "log.csv")
    log_data = pd.read_csv(log_filepath)

    labels_image_format_list = list()
    preds_image_format_list = list()
    preds_fixed_image_format_list = list()

    # Using iloc to select the first column
    for data_basename in tqdm(log_data.iloc[:, 0]):
        labels_image_format = os.path.join(input_folders["labels_2d"], f"{data_basename}_<VIEW>.png")
        pred_image_format = os.path.join(input_folders["preds_2d"], f"{data_basename}_<VIEW>.png")
        pred_fixed_image_format = os.path.join(input_folders["preds_fixed_2d"], f"{data_basename}_<VIEW>.png")

        labels_image_format_list.append(labels_image_format)
        preds_image_format_list.append(pred_image_format)
        preds_fixed_image_format_list.append(pred_fixed_image_format)

    filepaths_count = len(labels_image_format_list)
    for filepath_idx in tqdm(range(filepaths_count)):
        # Get index data
        label_image_format = labels_image_format_list[filepath_idx]
        pred_image_format = preds_image_format_list[filepath_idx]
        pred_fixed_image_format = preds_fixed_image_format_list[filepath_idx]

        # Label
        label_image_relative = pathlib.Path(label_image_format).relative_to(input_folders["labels_2d"])
        label_image_output_filepath = os.path.join(output_folders["labels_3d_reconstruct"], label_image_relative)

        # Pred
        pred_image_relative = pathlib.Path(pred_image_format).relative_to(input_folders["preds_2d"])
        pred_image_output_filepath = os.path.join(output_folders["preds_3d_reconstruct"], pred_image_relative)

        # Pred Fixed
        pred_fixed_image_relative = pathlib.Path(pred_fixed_image_format).relative_to(input_folders["preds_fixed_2d"])
        pred_fixed_image_output_filepath = os.path.join(output_folders["preds_fixed_3d_reconstruct"], pred_fixed_image_relative)

        # 3D Folders - labels
        save_filename = label_image_output_filepath.replace("_<VIEW>.png", "")
        label_3d_path = pathlib.Path(source_folders["labels_3d"])
        label_3d_relative = str(label_image_relative).replace("_<VIEW>.png", "*")
        source_label_3d_data_filepath = list(label_3d_path.rglob(label_3d_relative))[0]

        label_numpy_data = reconstruct_3d_from_2d(
            format_of_2d_images=label_image_format,
            source_data_filepath=source_label_3d_data_filepath
        )
        convert_numpy_to_data_file(
            numpy_data=label_numpy_data,
            source_data_filepath=source_label_3d_data_filepath,
            save_filename=save_filename
        )

        # 3D Folders - preds
        save_filename = pred_image_output_filepath.replace("_<VIEW>.png", "")
        pred_3d_path = pathlib.Path(source_folders["preds_3d"])
        pred_3d_relative = str(pred_image_relative).replace("_<VIEW>.png", "*")
        source_pred_3d_data_filepath = list(pred_3d_path.rglob(pred_3d_relative))[0]

        pred_numpy_data = reconstruct_3d_from_2d(
            format_of_2d_images=pred_image_format,
            source_data_filepath=source_pred_3d_data_filepath
        )
        convert_numpy_to_data_file(
            numpy_data=pred_numpy_data,
            source_data_filepath=source_pred_3d_data_filepath,
            save_filename=save_filename
        )

        # 3D Folders - preds fixed
        save_filename = pred_fixed_image_output_filepath.replace("_<VIEW>.png", "")
        pred_fixed_3d_path = pathlib.Path(source_folders["preds_fixed_3d"])
        pred_fixed_3d_relative = str(pred_fixed_image_relative).replace("_<VIEW>.png", "*")
        source_pred_fixed_3d_data_filepath = list(pred_fixed_3d_path.rglob(pred_fixed_3d_relative))[0]

        pred_fixed_numpy_data = reconstruct_3d_from_2d(
            format_of_2d_images=pred_fixed_image_format,
            source_data_filepath=source_pred_fixed_3d_data_filepath
        )
        convert_numpy_to_data_file(
            numpy_data=pred_fixed_numpy_data,
            source_data_filepath=source_pred_fixed_3d_data_filepath,
            save_filename=save_filename
        )

    # format_of_2d_images = r".\parse_labels_mini_cropped_v5\PA000005_vessel_02584_<VIEW>.png"
    # final_data_3d = reconstruct_3d_from_2d(format_of_2d_images)
    #
    # voxel_grid = final_data_3d.astype(np.uint8)
    # refine_construction(voxel_grid)


# TODO: implement
def create_3d_fusions():
    # Inputs
    input_folders = {
        "labels_3d_reconstruct": os.path.join(TRAIN_CROPPED_PATH, "labels_3d_reconstruct_v6"),
        "preds_3d": os.path.join(TRAIN_CROPPED_PATH, "preds_3d_v6"),
        "preds_fixed_3d": os.path.join(TRAIN_CROPPED_PATH, "preds_fixed_3d_v6")
    }

    # Outputs
    output_folders = {
        "preds_3d_fusion": os.path.join(TRAIN_CROPPED_PATH, "preds_3d_fusion_v6"),
        "preds_fixed_3d_fusion": os.path.join(TRAIN_CROPPED_PATH, "preds_fixed_3d_fusion_v6")
    }

    # Create Output Folders
    for output_folder in output_folders.values():
        os.makedirs(output_folder, exist_ok=True)

    # Get the filepaths
    input_filepaths = dict()
    for key, value in input_folders.items():
        input_filepaths[key] = sorted(pathlib.Path(value).rglob("*.*"))

    filepaths_count = len(input_filepaths["labels_3d_reconstruct"])
    for filepath_idx in tqdm(range(filepaths_count)):
        # Get index data
        label_3d_reconstruct_filepath = input_filepaths["labels_3d_reconstruct"][filepath_idx]
        pred_3d_filepath = input_filepaths["preds_3d"][filepath_idx]
        pred_fixed_3d_filepath = input_filepaths["preds_fixed_3d"][filepath_idx]

        # Original data
        label_3d_reconstruct_numpy_data = convert_data_file_to_numpy(data_filepath=label_3d_reconstruct_filepath)
        pred_3d_numpy_data = convert_data_file_to_numpy(data_filepath=pred_3d_filepath)
        pred_fixed_3d_numpy_data = convert_data_file_to_numpy(data_filepath=pred_fixed_3d_filepath)

        # Fusions
        pred_3d_fusion = np.logical_or(pred_3d_numpy_data, label_3d_reconstruct_numpy_data)
        pred_3d_fusion = pred_3d_fusion.astype(np.float32)
        pred_fixed_3d_fusion = np.logical_or(pred_fixed_3d_numpy_data, label_3d_reconstruct_numpy_data)
        pred_fixed_3d_fusion = pred_fixed_3d_fusion.astype(np.float32)

        # Pred
        pred_3d_relative = pathlib.Path(pred_3d_filepath).relative_to(input_folders["preds_3d"])
        pred_3d_output_filepath = os.path.join(output_folders["preds_3d_fusion"], pred_3d_relative)

        # Pred Fixed
        pred_fixed_3d_relative = pathlib.Path(pred_fixed_3d_filepath).relative_to(input_folders["preds_fixed_3d"])
        pred_fixed_3d_output_filepath = os.path.join(output_folders["preds_fixed_3d_fusion"], pred_fixed_3d_relative)

        # 3D Folders - preds fusion
        save_filename = pred_3d_output_filepath
        convert_numpy_to_data_file(
            numpy_data=pred_3d_fusion,
            source_data_filepath=pred_3d_filepath,
            save_filename=save_filename
        )

        # 3D Folders - preds fixed
        save_filename = pred_fixed_3d_output_filepath
        convert_numpy_to_data_file(
            numpy_data=pred_fixed_3d_fusion,
            source_data_filepath=pred_fixed_3d_filepath,
            save_filename=save_filename
        )


def test_2d_to_3d_and_back(data_3d_filepath, cropped_data_path):
    # Use the 2d projections to create a 3d reconstruction
    # Use the 3d reconstruction to create a 2d projection
    # Compare the new 2d projection with the original 2d projection
    data_3d_basename = get_data_file_stem(data_filepath=data_3d_filepath)
    format_of_2d_images = os.path.join(cropped_data_path, f"{data_3d_basename}_<VIEW>.png")

    # Get the locally saved 2D images
    data_list = get_images_6_views(
        format_of_2d_images=format_of_2d_images,
        convert_to_3d=False,
        source_data_filepath=None
    )

    # Reconstruct 3D from 2D and Project 3D to 2D again
    projection_options = {
        "front": True,
        "back": True,
        "top": True,
        "bottom": True,
        "left": True,
        "right": True
    }

    data_3d = reconstruct_3d_from_2d(
        format_of_2d_images=format_of_2d_images,
        source_data_filepath=data_3d_filepath
    )
    projections = project_3d_to_2d(
        data_3d=data_3d,
        projection_options=projection_options,
        source_data_filepath=data_3d_filepath
    )

    # Compare the 2D projections with the original 2D images
    equal_status = True
    for image_idx, image_view in tqdm(enumerate(IMAGES_6_VIEWS)):
        locally_saved_image = data_list[image_idx]
        projection_image = projections[f"{image_view}_image"]
        if (locally_saved_image - projection_image).sum() != 0:
            equal_status = False
            print(f"Error in '{image_view}' view")

    if equal_status is True:
        print("All images are equal")


def main():
    create_3d_reconstructions()
    create_3d_fusions()

    # TODO: DEBUG
    # data_3d_filepath = os.path.join(TRAIN_CROPPED_PATH, "labels_3d_v6", "PA000005_11899.nii.gz")
    # cropped_data_path = os.path.join(TRAIN_CROPPED_PATH, "labels_2d_v6")
    # test_2d_to_3d_and_back(data_3d_filepath=data_3d_filepath, cropped_data_path=cropped_data_path)


if __name__ == "__main__":
    # TODO:
    # 1. Make sure DATASET_FOLDER folder is present in the root directory
    main()
