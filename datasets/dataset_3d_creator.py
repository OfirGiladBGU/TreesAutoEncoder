import numpy as np
import cv2
import os

from dataset_list import DATA_PATH
from dataset_utils import convert_numpy_to_nii_gz, reverse_rotations


def reconstruct_3d_from_2d(format_of_2d_images):
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

def full_3d_reconstruction(src_folder, tgt_folder):
    os.makedirs(tgt_folder, exist_ok=True)

    all_images = os.listdir(src_folder)
    image_format_set = set()
    for image in all_images:
        image_split = image.rsplit("_", 1)
        image_split[1] = "<VIEW>.png"
        image_format = "_".join(image_split)
        image_format_set.add(image_format)

    image_format_set = list(image_format_set)
    for image_format in image_format_set:
        image_format_src_filepath = os.path.join(src_folder, image_format)
        final_data_3d = reconstruct_3d_from_2d(format_of_2d_images=image_format_src_filepath)

        save_name = image_format.replace("<VIEW>.png", "result")
        print(f"Saving: {save_name}.nii.gz")
        save_name = os.path.join(tgt_folder, save_name)
        convert_numpy_to_nii_gz(final_data_3d, save_name=save_name)

    # format_of_2d_images = r".\parse_labels_mini_cropped_v5\PA000005_vessel_02584_<VIEW>.png"
    # final_data_3d = reconstruct_3d_from_2d(format_of_2d_images)
    #
    # voxel_grid = final_data_3d.astype(np.uint8)
    # refine_construction(voxel_grid)


def main():
    src_folder1 = os.path.join(DATA_PATH, "parse_labels_mini_cropped_v5")
    tgt_folder1 = os.path.join(DATA_PATH, "parse_labels_mini_cropped_3d_reconstruct_v5")

    # src_folder2 = os.path.join(DATA_PATH, "parse_preds_mini_cropped_v5")
    # tgt_folder2 = os.path.join(DATA_PATH, "parse_preds_mini_cropped_3d_reconstruct_v5")

    full_3d_reconstruction(src_folder1, tgt_folder1)
    # full_3d_reconstruction(src_folder2, tgt_folder2)


if __name__ == "__main__":
    # TODO:
    # 1. Make sure "parse2022" folder is present in the root directory
    main()
