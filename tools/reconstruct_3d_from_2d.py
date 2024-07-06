import numpy as np
import nibabel as nib
import cv2
import os


def convert_numpy_to_nii_gz(numpy_array, save_name="", save=False):
    ct_nii_gz = nib.Nifti1Image(numpy_array, affine=np.eye(4))
    if save and save_name != "":
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

    final_data_3d = final_data_3d.astype(np.uint8) * 255
    save_name = format_of_2d_images.replace("<VIEW>", "result")
    convert_numpy_to_nii_gz(final_data_3d, save_name=save_name, save=True)


def main():
    format_of_2d_images = r".\parse_preds_mini_cropped_v3\PA000005_vessel_03253_<VIEW>.png"
    reconstruct_3d_from_2d(format_of_2d_images)


if __name__ == "__main__":
    # TODO:
    # 1. Make sure "parse2022" folder is present in the root directory
    main()
