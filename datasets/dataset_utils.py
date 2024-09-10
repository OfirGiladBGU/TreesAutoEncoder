import numpy as np
import nibabel as nib
import torch


def convert_nii_gz_to_numpy(data_filepath) -> np.ndarray:
    nib_data = nib.load(data_filepath)
    numpy_data = nib_data.get_fdata()
    return numpy_data


def convert_numpy_to_nii_gz(numpy_data: np.ndarray, save_name=None) -> nib.Nifti1Image:
    nib_data = nib.Nifti1Image(numpy_data, affine=np.eye(4))
    if save_name is not None:
        if not save_name.endswith(".nii.gz"):
            save_name = f"{save_name}.nii.gz"
        nib.save(img=nib_data, filename=save_name)
    return nib_data


def reverse_rotations(numpy_image: np.ndarray, view_type: str) -> np.ndarray:
    # Convert to 3D
    data_3d = np.zeros(shape=(numpy_image.shape[0], numpy_image.shape[0], numpy_image.shape[0]), dtype=np.uint8)
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


def apply_threshold(tensor: torch.Tensor, threshold: float):
    tensor[tensor >= threshold] = 1.0
    tensor[tensor < threshold] = 0.0
