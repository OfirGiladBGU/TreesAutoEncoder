import numpy as np
import cv2
import os
import pathlib
from tqdm import tqdm

from dataset_list import CROPPED_PATH
from dataset_utils import convert_numpy_to_nii_gz, reverse_rotations


# TODO: Use Pred for Fusion
def reconstruct_3d_from_2d(format_of_2d_images) -> np.ndarray:
    images_6_views = ['top', 'bottom', 'front', 'back', 'left', 'right']
    data_3d_list = list()
    for image_view in images_6_views:
        image_path = format_of_2d_images.replace("<VIEW>", image_view)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        numpy_image = np.array(image)
        data_3d = reverse_rotations(numpy_image, image_view)
        data_3d_list.append(data_3d)

    final_data_3d = data_3d_list[0]
    for i in range(1, len(data_3d_list)):
        final_data_3d = np.logical_or(final_data_3d, data_3d_list[i])

    final_data_3d = final_data_3d.astype(np.float32)

    # save_name = format_of_2d_images.replace("<VIEW>", "result")
    # convert_numpy_to_nii_gz(final_data_3d, save_name=save_name)

    return final_data_3d


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


def full_3d_reconstruction():
    # Inputs
    input_folders = {
        "labels_2d": os.path.join(CROPPED_PATH, "labels_2d_v6"),
        "preds_2d": os.path.join(CROPPED_PATH, "preds_2d_v6"),
        "preds_fixed_2d": os.path.join(CROPPED_PATH, "preds_fixed_2d_v6")
    }

    # Outputs
    output_folders = {
        "labels_3d_reconstruct": os.path.join(CROPPED_PATH, "labels_3d_reconstruct_v6"),
        "preds_3d_reconstruct": os.path.join(CROPPED_PATH, "preds_3d_reconstruct_v6"),
        "preds_fixed_3d_reconstruct": os.path.join(CROPPED_PATH, "preds_fixed_3d_reconstruct_v6")
    }

    # Create Output Folders
    for output_folder in output_folders.values():
        os.makedirs(output_folder, exist_ok=True)

    # Get the filepaths
    input_filepaths = dict()
    for key, value in input_folders.items():
        input_filepaths[key] = sorted(pathlib.Path(value).rglob("*.png"))

    labels_image_format_set = set()
    preds_image_format_set = set()
    preds_fixed_image_format_set = set()

    filepaths_count = len(input_filepaths["labels_2d"])
    for filepath_idx in tqdm(range(filepaths_count)):
        # Get index data:
        label_2d_filepath = input_filepaths["labels_2d"][filepath_idx]
        pred_2d_filepath = input_filepaths["preds_2d"][filepath_idx]
        pred_fixed_2d_filepath = input_filepaths["preds_fixed_2d"][filepath_idx]

        # Label
        label_image_split = str(label_2d_filepath).rsplit("_", 1)
        label_image_split[1] = "<VIEW>.png"
        label_image_format = "_".join(label_image_split)
        labels_image_format_set.add(label_image_format)

        # Pred
        pred_image_split = str(pred_2d_filepath).rsplit("_", 1)
        pred_image_split[1] = "<VIEW>.png"
        pred_image_format = "_".join(pred_image_split)
        preds_image_format_set.add(pred_image_format)

        # Pred Fixed
        pred_fixed_image_split = str(pred_fixed_2d_filepath).rsplit("_", 1)
        pred_fixed_image_split[1] = "<VIEW>.png"
        pred_fixed_image_format = "_".join(pred_fixed_image_split)
        preds_fixed_image_format_set.add(pred_fixed_image_format)

    labels_image_format_set = list(labels_image_format_set)
    preds_image_format_set = list(preds_image_format_set)
    preds_fixed_image_format_set = list(preds_fixed_image_format_set)

    filepaths_count = len(labels_image_format_set)
    for filepath_idx in tqdm(range(filepaths_count)):
        # Get index data:
        label_image_format = labels_image_format_set[filepath_idx]
        pred_image_format = preds_image_format_set[filepath_idx]
        pred_fixed_image_format = preds_fixed_image_format_set[filepath_idx]

        label_numpy_data = reconstruct_3d_from_2d(format_of_2d_images=label_image_format)
        pred_numpy_data = reconstruct_3d_from_2d(format_of_2d_images=pred_image_format)
        pred_fixed_numpy_data = reconstruct_3d_from_2d(format_of_2d_images=pred_fixed_image_format)

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
        save_name = label_image_output_filepath.replace("_<VIEW>.png", "")
        convert_numpy_to_nii_gz(numpy_data=label_numpy_data, save_name=save_name)

        # 3D Folders - preds
        save_name = pred_image_output_filepath.replace("_<VIEW>.png", "")
        convert_numpy_to_nii_gz(numpy_data=pred_numpy_data, save_name=save_name)

        # 3D Folders - preds fixed
        save_name = pred_fixed_image_output_filepath.replace("_<VIEW>.png", "")
        convert_numpy_to_nii_gz(numpy_data=pred_fixed_numpy_data, save_name=save_name)

    # format_of_2d_images = r".\parse_labels_mini_cropped_v5\PA000005_vessel_02584_<VIEW>.png"
    # final_data_3d = reconstruct_3d_from_2d(format_of_2d_images)
    #
    # voxel_grid = final_data_3d.astype(np.uint8)
    # refine_construction(voxel_grid)


def main():
    full_3d_reconstruction()


if __name__ == "__main__":
    # TODO:
    # 1. Make sure "parse2022" folder is present in the root directory
    main()
