import numpy as np
import nibabel as nib
import cv2
import os


def convert_numpy_to_nii_gz(numpy_array, save_name=None):
    ct_nii_gz = nib.Nifti1Image(numpy_array, affine=np.eye(4))
    if save_name is not None:
        nib.save(ct_nii_gz, f"{save_name}.nii.gz")
    return ct_nii_gz


def reverse_rotations(numpy_image, view_type):
    # Convert to 3D
    data_3d = np.zeros((numpy_image.shape[0], numpy_image.shape[0], numpy_image.shape[0]), dtype=np.uint8)
    for i in range(numpy_image.shape[0]):
        for j in range(numpy_image.shape[1]):
            gray_value = int(numpy_image[i, j])
            if gray_value > 0:
                rescale_gray_value = int(numpy_image.shape[0] * (1 - (gray_value / 255)))

                if view_type in ["front", "back"]:
                    data_3d[i, j, rescale_gray_value] = 255
                elif view_type in ["top", "bottom"]:
                    data_3d[rescale_gray_value, i, j] = 255
                elif view_type in ["right", "left"]:
                    data_3d[i, rescale_gray_value, j] = 255
                else:
                    raise ValueError("Invalid view type")

    # Reverse the rotations
    if view_type == "front":
        pass

    if view_type == "back":
        data_3d = np.rot90(data_3d, k=2, axes=(2, 1))

    if view_type == "top":
        data_3d = np.rot90(data_3d, k=1, axes=(2, 1))

    if view_type == "bottom":
        data_3d = np.rot90(data_3d, k=2, axes=(1, 0))
        data_3d = np.rot90(data_3d, k=1, axes=(2, 1))

    if view_type == "right":
        data_3d = np.flip(data_3d, axis=1)

    if view_type == "left":
        data_3d = np.rot90(data_3d, k=2, axes=(2, 1))
        data_3d = np.flip(data_3d, axis=1)

    # Reverse the initial rotations
    data_3d = np.flip(data_3d, axis=1)
    data_3d = np.rot90(data_3d, k=1, axes=(2, 1))
    data_3d = np.rot90(data_3d, k=1, axes=(2, 0))

    return data_3d


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

    final_data_3d = final_data_3d.astype(np.float64)
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


def main():
    src_folder = r".\parse_labels_mini_cropped_v5"
    tgt_folder = r".\parse_labels_mini_cropped_3d_reconstruct_v5"
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
        save_name = os.path.join(tgt_folder, save_name)
        convert_numpy_to_nii_gz(final_data_3d, save_name=save_name)


    # format_of_2d_images = r".\parse_labels_mini_cropped_v5\PA000005_vessel_02584_<VIEW>.png"
    # final_data_3d = reconstruct_3d_from_2d(format_of_2d_images)
    #
    # voxel_grid = final_data_3d.astype(np.uint8)
    # refine_construction(voxel_grid)


if __name__ == "__main__":
    # TODO:
    # 1. Make sure "parse2022" folder is present in the root directory
    main()
