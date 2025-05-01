import os
import pathlib
import numpy as np
import torch
import cv2
from typing import Union, Dict, Tuple
from scipy.ndimage import label, binary_dilation, generate_binary_structure
from itertools import product

# For visualization
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from skimage import color

# For .nii.gz
import nibabel as nib

# For .ply and .obj
import trimesh
import open3d as o3d

from datasets_forge.dataset_configurations import IMAGES_6_VIEWS


###################
# Data Converters #
###################
def validate_data_paths(data_paths: list[str]):
    for data_path in data_paths:
        if not os.path.exists(data_path):
            raise ValueError(f"Invalid data path: {data_path}")
        elif len(os.listdir(data_path)) == 0:
            raise ValueError(f"Empty data path: {data_path}")
        else:
            pass


def get_data_file_extension(data_filepath) -> str:
    data_filepath = str(data_filepath)
    if data_filepath.endswith(".nii.gz"):
        data_extension = ".nii.gz"
    else:
        data_extension = pathlib.Path(data_filepath).suffix
    return data_extension


# TODO: Support for relative stem from the dataset folder (to support sub folders) - NEED TO TEST
def get_data_file_stem(data_filepath, relative_to=None) -> str:
    """
    :param data_filepath:
    :param relative_to: When provided, the stem will also contain the relative path to the data
    :return:
    """
    data_filepath = str(data_filepath)
    replace_extension = get_data_file_extension(data_filepath=data_filepath)
    if relative_to is not None:
        data_filepath_stem = os.path.relpath(data_filepath, relative_to)
    else:
        data_filepath_stem = os.path.basename(data_filepath)
    data_filepath_stem = data_filepath_stem.replace("\\", "/")  # Enable support for different OS
    data_filepath_stem = data_filepath_stem.replace(replace_extension, "")
    return data_filepath_stem


def convert_data_file_to_numpy(data_filepath, apply_data_threshold: bool = False, **kwargs) -> np.ndarray:
    if not os.path.exists(data_filepath):
        raise ValueError(f"Invalid data path: {data_filepath}")
    extension_map = {
        # 2D
        ".png": _convert_png_to_numpy,
        # 3D
        ".nii.gz": _convert_nii_gz_to_numpy,
        ".ply": _convert_ply_to_numpy,
        ".obj": _convert_obj_to_numpy,
        ".pcd": _convert_pcd_to_numpy,
        ".npy": _convert_npy_to_numpy
    }
    data_filepath = str(data_filepath)
    data_extension = get_data_file_extension(data_filepath=data_filepath)

    if data_extension in extension_map.keys():
        numpy_data = extension_map[data_extension](data_filepath=data_filepath, **kwargs)
        if apply_data_threshold is True:
            apply_threshold(numpy_data, threshold=0.5, keep_values=False)
        return numpy_data
    else:
        raise ValueError(f"Invalid data format (Got extension: '{data_extension}')")


def convert_numpy_to_data_file(numpy_data: np.ndarray, source_data_filepath, save_filename=None,
                               apply_data_threshold: bool = False, **kwargs):
    extension_map = {
        # 2D
        ".png": _convert_numpy_to_png,
        # 3D
        ".nii.gz": _convert_numpy_to_nii_gz,
        ".ply": _convert_numpy_to_ply,
        ".obj": _convert_numpy_to_obj,
        ".pcd": _convert_numpy_to_pcd,
        ".npy": _convert_numpy_to_npy  # Notice: Save as .npy ignores the source_data_filepath
    }
    source_data_filepath = str(source_data_filepath)
    data_extension = get_data_file_extension(data_filepath=source_data_filepath)

    if data_extension in extension_map.keys():
        if apply_data_threshold:
            apply_threshold(numpy_data, threshold=0.5, keep_values=False)
        return extension_map[data_extension](
            numpy_data=numpy_data,
            source_data_filepath=source_data_filepath,
            save_filename=save_filename,
            **kwargs
        )
    else:
        raise ValueError(f"Invalid data format (Got extension: '{data_extension}')")


#################################
# png to numpy and numpy to png #
#################################
def _convert_png_to_numpy(data_filepath: str, **kwargs) -> np.ndarray:
    numpy_data = cv2.imread(data_filepath)
    numpy_data = cv2.cvtColor(numpy_data, cv2.COLOR_BGR2GRAY)
    return numpy_data


def _convert_numpy_to_png(numpy_data: np.ndarray, source_data_filepath=None, save_filename=None,
                          **kwargs) -> np.ndarray:
    if save_filename is not None and len(save_filename) > 0:
        save_filename = str(save_filename)
        os.makedirs(name=os.path.dirname(save_filename), exist_ok=True)
        if not save_filename.endswith(".png"):
            save_filename = f"{save_filename}.png"
        cv2.imwrite(save_filename, numpy_data)  # Save to PNG

    return numpy_data

#######################################
# nii.gz to numpy and numpy to nii.gz #
#######################################

# TODO: return or apply the affine transformation to the numpy data for the save later
def _convert_nii_gz_to_numpy(data_filepath: str, **kwargs) -> np.ndarray:
    nifti_data = nib.load(data_filepath)
    numpy_data = nifti_data.get_fdata()
    return numpy_data


def _convert_numpy_to_nii_gz(numpy_data: np.ndarray, source_data_filepath=None, save_filename=None,
                             **kwargs) -> nib.Nifti1Image:
    if source_data_filepath is not None:
        nifti_data = nib.load(source_data_filepath)
        new_nifti_data = nib.Nifti1Image(numpy_data, affine=nifti_data.affine, header=nifti_data.header)
        if "int" in str(numpy_data.dtype):  # Keep integer type for components data
            new_nifti_data.header.set_data_dtype(np.int16) # Assumption: Max value will be less than 32767
    else:
        new_nifti_data = nib.Nifti1Image(numpy_data, affine=np.eye(4))

    if save_filename is not None and len(save_filename) > 0:
        save_filename = str(save_filename)
        os.makedirs(name=os.path.dirname(save_filename), exist_ok=True)
        if not save_filename.endswith(".nii.gz"):
            save_filename = f"{save_filename}.nii.gz"
        nib.save(img=new_nifti_data, filename=save_filename)  # Save to NII.GZ
    return new_nifti_data


# DEBUG: Save NII.GZ with identity affine
def save_nii_gz_in_identity_affine(numpy_data=None, data_filepath=None, save_filename=None,
                                   **kwargs) -> nib.Nifti1Image:
    if data_filepath is not None:
        nifti_data = nib.load(data_filepath)
        numpy_data = nifti_data.get_fdata()
    elif numpy_data is not None:
        pass
    else:
        raise ValueError("Provide either numpy data or data filepath")
    new_nifti_data = nib.Nifti1Image(numpy_data, affine=np.eye(4))

    if save_filename is not None and len(save_filename) > 0:
        save_filename = str(save_filename)
        os.makedirs(name=os.path.dirname(save_filename), exist_ok=True)
        if not save_filename.endswith(".nii.gz"):
            save_filename = f"{save_filename}.nii.gz"
        nib.save(img=new_nifti_data, filename=save_filename)  # Save to NII.GZ
    return new_nifti_data


#################################
# ply to numpy and numpy to ply #
#################################

# TODO: Check how to make Abstract converter from supported file formats

def _convert_ply_to_numpy(data_filepath: str, **kwargs) -> np.ndarray:
    # Mesh PLY
    if data_filepath.endswith("mesh.ply"):
        numpy_data = _convert_obj_to_numpy(data_filepath=data_filepath, **kwargs)
    # Point Cloud PLY
    elif data_filepath.endswith("pcd.ply"):
        numpy_data = _convert_pcd_to_numpy(data_filepath=data_filepath, **kwargs)
    else:
        raise ValueError("Invalid data format")

    return numpy_data


def _convert_numpy_to_ply(numpy_data: np.ndarray, source_data_filepath=None, save_filename=None,
                          **kwargs) -> Union[trimesh.Trimesh, o3d.geometry.PointCloud]:
    # Mesh PLY
    if source_data_filepath.endswith("mesh.ply"):
        ply_extension = "mesh.ply"
        mesh = _convert_numpy_to_obj(numpy_data=numpy_data, source_data_filepath=source_data_filepath, **kwargs)
        new_ply_data = mesh

    # Point Cloud PLY
    elif source_data_filepath.endswith("pcd.ply"):
        ply_extension = "pcd.ply"
        pcd = _convert_numpy_to_pcd(numpy_data=numpy_data, source_data_filepath=source_data_filepath, **kwargs)
        new_ply_data = pcd
    else:
        raise ValueError("Invalid data format")


    if save_filename is not None and len(save_filename) > 0:
        save_filename = str(save_filename)
        os.makedirs(name=os.path.dirname(save_filename), exist_ok=True)
        save_filename = f"{save_filename}_{ply_extension}"
        if "mesh" in ply_extension:
            new_ply_data.export(file_obj=save_filename)  # Save to PLY
        elif "pcd" in ply_extension:
            o3d.io.write_point_cloud(filename=save_filename, pointcloud=new_ply_data)  # Save to PLY
        else:
            raise ValueError("Invalid data format")

    return new_ply_data


#################################
# obj to numpy and numpy to obj #
#################################
def _convert_obj_to_numpy(data_filepath: str, **kwargs) -> np.ndarray:
    mesh_scale = kwargs.get("mesh_scale", 1.0)  # Define points scale
    voxel_size = kwargs.get("voxel_size", 2.0)  # Define voxel size (the size of each grid cell)

    mesh = trimesh.load(data_filepath)
    if mesh_scale != 1.0:
        mesh = mesh.apply_scale(mesh_scale)
    voxelized = mesh.voxelized(pitch=voxel_size)  # Pitch = voxel size

    numpy_data = voxelized.matrix.astype(np.uint8)
    return numpy_data


def _convert_numpy_to_obj(numpy_data: np.ndarray, source_data_filepath=None, save_filename=None,
                          **kwargs) -> trimesh.Trimesh:
    # V1
    # voxel_size = kwargs.get("voxel_size", 0.05)  # Define voxel size (the size of each grid cell)
    #
    # occupied_indices = np.argwhere(numpy_data == 1)  # Find occupied voxels (indices where numpy_data == 1)
    # centers = occupied_indices * voxel_size  # Convert indices to real-world coordinates (Scale by voxel size)
    #
    # # Create cube meshes for each occupied voxel
    # cubes = []
    # for center in centers:
    #     # Create a cube for each voxel
    #     cube = trimesh.creation.box(
    #         extents=[voxel_size] * 3,
    #         transform=trimesh.transformations.translation_matrix(center)
    #     )
    #     cubes.append(cube)
    # mesh = trimesh.util.concatenate(cubes)  # Combine all cubes into a single mesh

    # V2

    # TODO: Test

    mesh_scale = kwargs.get("mesh_scale", 1.0)  # Define points scale [Original]
    voxel_size = kwargs.get("voxel_size", 2.0)  # Define voxel size (the size of each grid cell) [Original]

    # Find minimum bounds
    if source_data_filepath != "dummy.obj":
        source_mesh = trimesh.load(source_data_filepath)
        min_bounds = source_mesh.bounds[0]  # Extract minimum bounds from source mesh
    else:
        min_bounds = np.array([0, 0, 0])

    occupied_indices = np.argwhere(numpy_data == 1.0)  # Find occupied voxels (indices where numpy_data == 1)
    centers = (occupied_indices + min_bounds)  # Apply shift (align with the original mesh space) and scale correctly

    # Create cube meshes for each occupied voxel
    cubes = []
    for center in centers:
        # Create a cube for each voxel
        cube = trimesh.creation.box(
            extents=[1, 1, 1],  # Voxel size is 1x1x1
            transform=trimesh.transformations.translation_matrix(center)
        )
        cubes.append(cube)
    mesh = trimesh.util.concatenate(cubes)  # Combine all cubes into a single mesh
    if voxel_size != 1.0:
        mesh.apply_scale(1.0 / voxel_size)  # Apply reverse the scale
    if mesh_scale != 1.0:
        mesh.apply_scale(1.0 / mesh_scale)  # Apply reverse the scale


    if save_filename is not None and len(save_filename) > 0:
        save_filename = str(save_filename)
        os.makedirs(name=os.path.dirname(save_filename), exist_ok=True)
        if not save_filename.endswith("obj"):
            save_filename = f"{save_filename}.obj"
        mesh.export(file_obj=save_filename)  # Save to OBJ

    new_obj_data = mesh
    return new_obj_data


#################################
# pcd to numpy and numpy to pcd #
#################################
def _convert_pcd_to_numpy(data_filepath: str, **kwargs) -> np.ndarray:
    # V1 - Using Open3D VoxelGrid
    points_scale = kwargs.get("points_scale", 1.0)  # Define points scale
    voxel_size = kwargs.get("voxel_size", 1.0)  # Define voxel size (the size of each grid cell)

    # Find voxel grid
    pcd = o3d.io.read_point_cloud(data_filepath)
    if points_scale != 1.0:
        pcd.scale(scale=points_scale, center=pcd.get_center())  # Scale relative to center
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(input=pcd, voxel_size=voxel_size)  # Voxelize pcd

    # Build numpy data
    grid_indices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])  # Get voxel centers
    grid_shape = np.max(grid_indices, axis=0) + 1
    numpy_data = np.zeros(shape=grid_shape, dtype=np.uint8)
    for grid_index in grid_indices:
        numpy_data[tuple(grid_index)] = 1  # 1 = occupied, 0 = empty


    # V2 - Convert Points to Discrete Voxels (Alternative)
    # pcd = o3d.io.read_point_cloud(data_filepath)
    # points = np.asarray(pcd.points)
    #
    # rounded_points = np.round(points).astype(int)  # Round the point coordinates to the nearest integer
    #
    # min_coords = rounded_points.min(axis=0)  # Compute the minimum coordinates to shift all points to the positive space
    # shifted_points = rounded_points - min_coords
    #
    # max_coords = shifted_points.max(axis=0)  # Determine the size of the voxel grid
    # numpy_data = np.zeros((max_coords[0] + 1, max_coords[1] + 1, max_coords[2] + 1), dtype=np.uint8)
    #
    # for point in shifted_points:  # Set voxels corresponding to points to 1 (white)
    #     x, y, z = point
    #     numpy_data[x, y, z] = 1

    return numpy_data


def _convert_numpy_to_pcd(numpy_data: np.ndarray, source_data_filepath=None, save_filename=None,
                          **kwargs) -> o3d.geometry.PointCloud:
    # V1 - Using Open3D VoxelGrid
    # voxel_size = kwargs.get("voxel_size", 2.0)  # Define voxel size (the size of each grid cell)
    #
    # occupied_indices = np.argwhere(numpy_data == 1)  # Find occupied voxels (indices where numpy_data == 1)
    # points = occupied_indices * voxel_size  # Convert indices to real-world coordinates (Scale by voxel size)
    #
    # pcd = o3d.geometry.PointCloud()  # Create Open3D PointCloud
    # pcd.points = o3d.utility.Vector3dVector(points)


    # V2 - Convert Points to Discrete Voxels
    # pcd_data = o3d.io.read_point_cloud(source_data_filepath)  # Load the original PCD file to retrieve the shift
    # pcd_data_points = np.asarray(pcd_data.points)
    # shift = np.floor(pcd_data_points.min(axis=0)).astype(int)  # Recompute the original shift
    #
    # voxel_indices = np.array(np.nonzero(numpy_data)).T  # Find the indices of all non-zero voxels [Shape: (N, 3)]
    #
    # original_points = voxel_indices + shift  # Apply the inverse shift to recover the original coordinates
    #
    # pcd = o3d.geometry.PointCloud()  # Convert the points to Open3D PointCloud
    # pcd.points = o3d.utility.Vector3dVector(original_points)


    # V3 - Using Open3D VoxelGrid and correct shift
    points_scale = kwargs.get("points_scale", 1.0)  # Define points scale [Original]
    voxel_size = kwargs.get("voxel_size", 1.0)  # Define voxel size (the size of each grid cell) [Original]

    # Find origin
    if source_data_filepath != "dummy.pcd":
        source_pcd = o3d.io.read_point_cloud(source_data_filepath)
        if points_scale != 1.0:
            source_pcd.scale(scale=points_scale, center=source_pcd.get_center())  # Scale relative to center
        source_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(input=source_pcd, voxel_size=voxel_size)  # Voxelize pcd
        source_origin = source_voxel_grid.origin
    else:
        source_origin = np.array([0, 0, 0])

    grid_indices = np.argwhere(numpy_data == 1.0)  # Find the indices of all non-zero voxels [Shape: (N, 3)]
    voxels = (grid_indices + source_origin)  # Apply the inverse shift to recover the original coordinates

    pcd = o3d.geometry.PointCloud()  # Convert the points to Open3D PointCloud
    pcd.points = o3d.utility.Vector3dVector(voxels)

    if voxel_size != 1.0:
        pcd.scale(scale=(1.0 / voxel_size), center=pcd.get_center())  # Scale relative to center
    if points_scale != 1.0:
        pcd.scale(scale=(1.0 / points_scale), center=pcd.get_center())  # Scale relative to center


    # Save the PCD
    if save_filename is not None and len(save_filename) > 0:
        save_filename = str(save_filename)
        os.makedirs(name=os.path.dirname(save_filename), exist_ok=True)
        if not save_filename.endswith(".pcd"):
            save_filename = f"{save_filename}.pcd"
        o3d.io.write_point_cloud(filename=save_filename, pointcloud=pcd, write_ascii=True)  # Save to PCD

    new_pcd_data = pcd
    return new_pcd_data


#################################
# npy to numpy and numpy to npy #
#################################
def _convert_npy_to_numpy(data_filepath: str, **kwargs) -> np.ndarray:
    numpy_data = np.load(file=data_filepath)
    return numpy_data


def _convert_numpy_to_npy(numpy_data: np.ndarray, source_data_filepath=None, save_filename=None,
                          **kwargs) -> np.ndarray:
    if save_filename is not None and len(save_filename) > 0:
        save_filename = str(save_filename)
        os.makedirs(name=os.path.dirname(save_filename), exist_ok=True)
        if not save_filename.endswith(".npy"):
            save_filename = f"{save_filename}.npy"
        np.save(file=save_filename, arr=numpy_data)  # Save to NPY
    return numpy_data


################
# Thresholding #
################
def apply_threshold(data: Union[torch.Tensor, np.ndarray], threshold: float, keep_values: bool = False):
    if keep_values is False:
        data[data >= threshold] = 1.0
    data[data < threshold] = 0.0


########################
# Connected Components #
########################
def connected_components_3d(data_3d: np.ndarray, connectivity_type: int = 26) -> Tuple[np.ndarray, int]:
    # Define the structure for connectivity
    # Here, we use a structure that connects each voxel to its immediate neighbors
    if connectivity_type == 26:
        structure = np.ones((3, 3, 3), dtype=np.int8)  # 26-connectivity

    elif connectivity_type == 6:
        structure = np.zeros((3, 3, 3), dtype=np.int8)
        active_points = [
            (0, 1, 1), (2, 1, 1),  # Points along the X-axis
            (1, 0, 1), (1, 2, 1),  # Points along the Y-axis
            (1, 1, 0), (1, 1, 2),  # Points along the Z-axis
            (1, 1, 1)  # Center point
        ]
        for x, y, z in active_points:
            structure[x, y, z] = 1

    else:
        raise ValueError("Invalid connectivity type")

    # Label connected components
    labeled_array, num_features = label(input=data_3d, structure=structure)

    # print("Labeled Array:", labeled_array)
    # print("Number of Features:", num_features)

    return labeled_array, num_features


def connected_components_2d(data_2d: np.ndarray, connectivity_type: int = 4) -> Tuple[np.ndarray, int]:
    if connectivity_type == 4:
        structure = None

    elif connectivity_type == 8:
        structure = np.ones(shape=(3, 3, 3), dtype=np.int8)

    else:
        raise ValueError("Invalid connectivity type")

    # Label connected components
    labeled_array, num_features = label(input=data_2d, structure=structure)

    # print("Labeled Array:", labeled_array)
    # print("Number of Features:", num_features)

    return labeled_array, num_features


#########################
# Components Continuity #
#########################

# Util #

def pad_data(numpy_data: np.ndarray, pad_width: int = 1) -> np.ndarray:
    return np.pad(numpy_data, pad_width=pad_width, mode='constant', constant_values=0)


def unpad_data(numpy_data: np.ndarray, pad_width: int = 1) -> np.ndarray:
    slices = tuple(slice(pad_width, -pad_width) for _ in range(numpy_data.ndim))
    return numpy_data[slices]


# # Inplace update
# def zero_corners_2d(numpy_data: np.ndarray):
#     # Best suited for cases where minimal connectivity type was set.
#
#     # Generate all corner coordinates: positions where each index is either 0 or -1
#     corners = list(product([0, -1], repeat=numpy_data.ndim))
#
#     for corner in corners:
#         numpy_data[corner] = 0


def local_scope_mask(block_mask: np.ndarray):
    """
    Expands the block mask to include face-connected neighbors (6-connectivity).

    Parameters:
        block_mask (ndarray): A binary mask of the cube (same shape as original array).

    Returns:
        ndarray: A boolean mask of the scope (including the original block).
    """
    # Generate 6-connected structure (face connectivity)
    struct = generate_binary_structure(rank=block_mask.ndim, connectivity=1)

    # Dilate to include neighbors
    expanded_mask = binary_dilation(block_mask, structure=struct)

    return expanded_mask  # block + its face neighbors


def get_local_scope_mask(numpy_data: np.ndarray, padding_size: int):
    block_mask = np.ones_like(a=numpy_data)
    block_mask = unpad_data(numpy_data=block_mask, pad_width=padding_size)
    block_mask = pad_data(numpy_data=block_mask, pad_width=padding_size)
    expand_mask = local_scope_mask(block_mask=block_mask)
    return expand_mask


def apply_local_scope_mask(numpy_data: np.ndarray, expand_mask: np.ndarray):
    numpy_data = np.where(expand_mask > 0.0, numpy_data, 0.0)


# 3D #

def components_continuity_3d_single_component(label_cube: np.ndarray, pred_advanced_fixed_cube: np.ndarray,
                                              reverse_mode: bool = False,
                                              connectivity_type: int = 26,
                                              delta_mode: int = 0,
                                              hard_condition: bool = False) -> np.ndarray:
    if delta_mode == 0:
        # Calculate the missing connected components in preds fixed
        pred_advanced_fixed_binary = pred_advanced_fixed_cube.astype(np.uint8)

        # Compare pixels mask
        label_binary = label_cube.astype(np.uint8)
        delta_cube = np.logical_xor(label_binary, pred_advanced_fixed_binary).astype(np.uint8)
        # delta_cube = ((label_cube - pred_advanced_fixed_cube) > 0.5).astype(np.uint8)

        # Identify connected components in delta_cube
        delta_labeled, delta_num_components = connected_components_3d(
            data_3d=delta_cube,
            connectivity_type=connectivity_type
        )
    else: # Custom mode for [Predict Pipeline]
        pred_advanced_fixed_binary = label_cube.astype(np.uint8)

        delta_labeled, delta_num_components = connected_components_3d(
            data_3d=pred_advanced_fixed_cube,
            connectivity_type=connectivity_type
        )

    # Iterate through connected components in delta_cube
    for component_label in range(1, delta_num_components + 1):
        # Create a mask for the current connected component
        component_mask = np.equal(delta_labeled, component_label).astype(np.uint8)

        # Check the number of connected components before adding the mask
        (_, components_before) = connected_components_3d(data_3d=pred_advanced_fixed_binary, connectivity_type=connectivity_type)
        # (_, components_before) = connected_components_3d(data_3d=pred_advanced_fixed_binary, connectivity_type=6)

        # Create a temporary data with the component added
        temp_fixed = np.logical_or(pred_advanced_fixed_binary, component_mask)
        (_, components_after) = connected_components_3d(data_3d=temp_fixed, connectivity_type=connectivity_type)
        # (_, components_after) = connected_components_3d(data_3d=temp_fixed, connectivity_type=6)

        if reverse_mode is False:
            # Add the component only if it does not decrease the number of connected components [Dataset Creation]
            condition = not (components_before > components_after)
        else:
            # Add the component only if it does not increase the number of connected components [Predict Pipeline]
            if hard_condition is True:
                condition = not (components_before <= components_after)
            else:
                condition = not (components_before < components_after)

        if condition:
            pred_advanced_fixed_binary = temp_fixed
        else:
            # print("Debug")
            pass

    # Update the pred_advanced_fixed_cube
    pred_advanced_fixed_cube = pred_advanced_fixed_binary.astype(pred_advanced_fixed_cube.dtype)

    return pred_advanced_fixed_cube


def components_continuity_3d_local_connectivity(label_cube: np.ndarray, pred_advanced_fixed_cube: np.ndarray,
                                                reverse_mode: bool = False,
                                                connectivity_type: int = 26,
                                                hard_condition: bool = True) -> np.ndarray:
    # padding_size = 2
    padding_size = 1

    # Calculate the missing connected components in preds fixed
    padded_pred_advanced_fixed = pad_data(numpy_data=pred_advanced_fixed_cube, pad_width=padding_size)
    pred_advanced_fixed_binary = padded_pred_advanced_fixed.astype(np.uint8)

    # Compare pixels mask
    padded_label = pad_data(numpy_data=label_cube, pad_width=padding_size)
    label_binary = padded_label.astype(np.uint8)
    delta_cube = np.logical_xor(label_binary, pred_advanced_fixed_binary).astype(np.uint8)
    # delta_cube = ((label_cube - pred_advanced_fixed_cube) > 0.5).astype(np.uint8)

    # Identify connected components in delta_cube
    delta_labeled, delta_num_components = connected_components_3d(
        data_3d=delta_cube,
        connectivity_type=connectivity_type
    )

    # Iterate through connected components in delta_cube
    for component_label in range(1, delta_num_components + 1):
        # Create a mask for the current connected component
        component_mask = np.equal(delta_labeled, component_label).astype(np.uint8)

        # ROI - cropped area between the component mask: top, bottom, left, right, front, back
        coords = np.argwhere(component_mask > 0)

        # Get bounding box in 3D
        top = np.min(coords[:, 0])  # Minimum row index (Y-axis)
        bottom = np.max(coords[:, 0])  # Maximum row index (Y-axis)

        left = np.min(coords[:, 1])  # Minimum column index (X-axis)
        right = np.max(coords[:, 1])  # Maximum column index (X-axis)

        front = np.min(coords[:, 2])  # Minimum depth index (Z-axis)
        back = np.max(coords[:, 2])  # Maximum depth index (Z-axis)

        # Ensure ROI is within valid bounds
        min_y = top - padding_size
        max_y = bottom + padding_size + 1

        min_x = left - padding_size
        max_x = right + padding_size + 1

        min_z = front - padding_size
        max_z = back + padding_size + 1

        # Check the number of connected components before adding the mask
        roi_temp_before = pred_advanced_fixed_binary[min_y:max_y, min_x:max_x, min_z:max_z]
        expand_mask = get_local_scope_mask(numpy_data=roi_temp_before, padding_size=padding_size)
        apply_local_scope_mask(numpy_data=roi_temp_before, expand_mask=expand_mask)
        (_, components_before) = connected_components_3d(data_3d=roi_temp_before, connectivity_type=connectivity_type)
        # (_, components_before) = connected_components_3d(data_3d=roi_temp_before, connectivity_type=6)

        # Create a temporary data with the component added
        temp_fix = np.logical_or(pred_advanced_fixed_binary, component_mask)
        roi_temp_after = temp_fix[min_y:max_y, min_x:max_x, min_z:max_z]
        apply_local_scope_mask(numpy_data=roi_temp_after, expand_mask=expand_mask)
        (_, components_after) = connected_components_3d(data_3d=roi_temp_after, connectivity_type=connectivity_type)
        # (_, components_after) = connected_components_3d(data_3d=roi_temp_after, connectivity_type=6)

        if reverse_mode is False:
            # Add the component only if it does not decrease the number of connected components
            # (on the local scope) [Dataset Creation]
            condition = not (components_before > components_after)
        else:
            # Add the component only if it does not increase the number of connected components
            # (on the local scope) [Predict Pipeline]
            if hard_condition is True:
                condition = not (components_before <= components_after)
            else:
                condition = not (components_before < components_after)

        if condition:
            pred_advanced_fixed_binary = temp_fix
        else:
            # print("Debug")
            pass

    # Update the pred_advanced_fixed_cube
    pred_advanced_fixed_binary = unpad_data(numpy_data=pred_advanced_fixed_binary, pad_width=padding_size)
    pred_advanced_fixed_cube = pred_advanced_fixed_binary.astype(pred_advanced_fixed_cube.dtype)
    return pred_advanced_fixed_cube


# 2D #

def components_continuity_2d_single_component(label_image: np.ndarray, pred_advanced_fixed_image: np.ndarray,
                                              reverse_mode: bool = False,
                                              connectivity_type: int = 4,
                                              binary_diff: bool = False,
                                              hard_condition: bool = True) -> np.ndarray:
    # Calculate the missing connected components in preds fixed
    pred_advanced_fixed_binary = (pred_advanced_fixed_image > 0).astype(np.uint8)

    if binary_diff is False:
        # Compare pixels values (Revealed occluded object behind a hole will be detected)
        delta_binary = ((label_image - pred_advanced_fixed_image) >= 1.0).astype(np.uint8)
    else:
        # Compare pixels mask (Revealed occluded object behind a hole will be ignored)
        label_binary = (label_image > 0).astype(np.uint8)
        delta_binary = np.logical_xor(label_binary, pred_advanced_fixed_binary).astype(np.uint8)
        # delta_binary = ((label_binary - pred_advanced_fixed_binary) > 0.5).astype(np.uint8)

    # Identify connected components in delta_binary
    delta_labeled, delta_num_components = connected_components_2d(
        data_2d=delta_binary,
        connectivity_type=connectivity_type
    )

    # Iterate through connected components in delta_binary
    for component_label in range(1, delta_num_components + 1):
        # Create a mask for the current connected component
        component_mask = np.equal(delta_labeled, component_label).astype(np.uint8)

        # Check the number of connected components before adding the mask
        (_, components_before) = connected_components_2d(data_2d=pred_advanced_fixed_binary, connectivity_type=connectivity_type)

        # Create a temporary image with the component added
        temp_fix = np.logical_or(pred_advanced_fixed_binary, component_mask)
        (_, components_after) = connected_components_2d(data_2d=temp_fix, connectivity_type=connectivity_type)

        if reverse_mode is False:
            # Add the component only if it does not decrease the number of connected components [Dataset Creation]
            condition = not (components_before > components_after)
        else:
            # Add the component only if it does not increase the number of connected components [Predict Pipeline]
            if hard_condition is True:
                condition = not (components_before <= components_after)
            else:
                condition = not (components_before < components_after)

        if condition:
            pred_advanced_fixed_binary = temp_fix
        else:
            # print("Debug")
            pass

    # Update the pred_advanced_fixed_image
    if reverse_mode is False:
        # Keep revealed occluded object in a hole [Dataset Creation]
        # pred_advanced_fixed_image = np.where(pred_advanced_fixed_binary > 0, pred_advanced_fixed_image, 0.0)
        # pred_advanced_fixed_image = np.where(pred_advanced_fixed_binary > 0, label_image, 0.0) # Keep occluded

        # TODO: Test
        # mask = pred_advanced_fixed_binary > 0
        # pred_advanced_fixed_image[mask] = np.where(
        #     pred_advanced_fixed_image[mask] > 0,
        #     pred_advanced_fixed_image[mask],
        #     label_image[mask]
        # )

        mask = pred_advanced_fixed_binary > 0
        pred_advanced_fixed_image[mask] = label_image[mask]
    else:
        # Remove outliers [Predict Pipeline]
        # pred_advanced_fixed_image = np.where(pred_advanced_fixed_binary > 0, label_image, 0.0)

        mask = pred_advanced_fixed_binary > 0
        pred_advanced_fixed_image[mask] = label_image[mask]

    return pred_advanced_fixed_image


def components_continuity_2d_local_connectivity(label_image: np.ndarray, pred_advanced_fixed_image: np.ndarray,
                                                reverse_mode: bool = False,
                                                connectivity_type: int = 4,
                                                binary_diff: bool = False,
                                                hard_condition: bool = True) -> np.ndarray:
    padding_size = 1

    # Calculate the missing connected components in preds fixed
    padded_label = pad_data(numpy_data=label_image, pad_width=padding_size)
    padded_pred_advanced_fixed = pad_data(numpy_data=pred_advanced_fixed_image, pad_width=padding_size)
    pred_advanced_fixed_binary = (padded_pred_advanced_fixed > 0).astype(np.uint8)

    if binary_diff is False:
        # Compare pixels values (Revealed occluded object behind a hole will be detected)
        delta_binary = ((padded_label - padded_pred_advanced_fixed) >= 1.0).astype(np.uint8)
    else:
        # Compare pixels mask (Revealed occluded object behind a hole will be ignored)
        label_binary = (padded_label > 0).astype(np.uint8)
        delta_binary = np.logical_xor(label_binary, pred_advanced_fixed_binary).astype(np.uint8)

    # Identify connected components in delta_binary
    delta_labeled, delta_num_components = connected_components_2d(
        data_2d=delta_binary,
        connectivity_type=connectivity_type
    )

    # Iterate through connected components in delta_binary
    for component_label in range(1, delta_num_components + 1):
        # Create a mask for the current connected component
        component_mask = np.equal(delta_labeled, component_label).astype(np.uint8)

        # ROI - cropped area between the component mask: top, bottom, left, right
        coords = np.argwhere(component_mask > 0)

        # Get bounding box
        top = np.min(coords[:, 0])  # Minimum row index
        bottom = np.max(coords[:, 0])  # Maximum row index

        left = np.min(coords[:, 1])  # Minimum column index
        right = np.max(coords[:, 1])  # Maximum column index

        # Ensure ROI is within valid bounds
        min_y = top - padding_size
        max_y = (bottom + 1) + padding_size

        min_x = left - padding_size
        max_x = (right + 1) + padding_size

        # Check the number of connected components before adding the mask
        roi_temp_before = pred_advanced_fixed_binary[min_y:max_y, min_x:max_x]
        expand_mask = get_local_scope_mask(numpy_data=roi_temp_before, padding_size=padding_size)
        apply_local_scope_mask(numpy_data=roi_temp_before, expand_mask=expand_mask)
        (_, components_before) = connected_components_2d(data_2d=roi_temp_before, connectivity_type=connectivity_type)

        # Create a temporary image with the component added
        temp_fix = np.logical_or(pred_advanced_fixed_binary, component_mask)
        roi_temp_after = temp_fix[min_y:max_y, min_x:max_x]
        apply_local_scope_mask(numpy_data=roi_temp_after, expand_mask=expand_mask)
        (_, components_after) = connected_components_2d(data_2d=roi_temp_after, connectivity_type=connectivity_type)

        if reverse_mode is False:
            # Add the component only if it does not decrease the number of connected components
            # (on the local scope) [Dataset Creation]
            condition = not (components_before > components_after)
        else:
            # Add the component only if it does not increase the number of connected components
            # (on the local scope) [Predict Pipeline]
            if hard_condition is True:
                condition = not (components_before <= components_after)
            else:
                condition = not (components_before < components_after)

        if condition:
            pred_advanced_fixed_binary = temp_fix
        else:
            # print("Debug")
            pass

    # Update the pred_advanced_fixed_image
    pred_advanced_fixed_binary = unpad_data(numpy_data=pred_advanced_fixed_binary, pad_width=padding_size)
    if reverse_mode is False:
        # Keep revealed occluded object in a hole [Dataset Creation]
        # pred_advanced_fixed_image = np.where(pred_advanced_fixed_binary > 0, pred_advanced_fixed_image, 0.0)

        # TODO: Test
        # mask = pred_advanced_fixed_binary > 0
        # pred_advanced_fixed_image[mask] = np.where(
        #     pred_advanced_fixed_image[mask] > 0,
        #     pred_advanced_fixed_image[mask],
        #     label_image[mask]
        # )

        mask = pred_advanced_fixed_binary > 0
        pred_advanced_fixed_image[mask] = label_image[mask]

        # pred_advanced_fixed_image = np.where(pred_advanced_fixed_binary > 0, label_image, 0.0) # Keep occluded
    else:
        # Remove outliers [Predict Pipeline]
        # pred_advanced_fixed_image = np.where(pred_advanced_fixed_binary > 0, label_image, 0.0)

        mask = pred_advanced_fixed_binary > 0
        pred_advanced_fixed_image[mask] = label_image[mask]

    return pred_advanced_fixed_image


#################
# 3D transforms #
#################

# TODO: Edit to support for cubes
def apply_rotations(data_3d: np.ndarray,
                    data_rotation: np.ndarray,
                    reverse: bool = False) -> np.ndarray:
    if np.array_equal(data_rotation, np.identity(3)):
        return data_3d
    else:
        # Ensure the rotation matrix is valid
        if data_rotation.shape != (3, 3):
            raise ValueError("Rotation matrix must be 3x3.")

        # Extract the indices of points with the target value
        points = np.argwhere(data_3d > 0)  # Nx3 array of coordinates

        if points.size == 0:
            # No points to rotate
            return np.zeros_like(data_3d)

        # Compute the center of the array
        center = np.array(data_3d.shape) / 2

        # Translate points to the center
        points_centered = points - center

        # Apply reverse rotation if specified
        if reverse:
            data_rotation = data_rotation.T  # Transpose is the inverse for a rotation matrix

        # Rotate the centered points
        rotated_points_centered = points_centered @ data_rotation.T  # Nx3 array

        # Translate points back to the original coordinate space
        rotated_points = rotated_points_centered + center

        # Round and cast to integer for indexing
        rotated_points = np.round(rotated_points).astype(int)

        # Create a new 3D array for the output
        output_array = np.zeros_like(data_3d)

        # Map rotated points back into the array, ensuring they are within bounds
        for point in rotated_points:
            if (0 <= point[0] < data_3d.shape[0] and
                    0 <= point[1] < data_3d.shape[1] and
                    0 <= point[2] < data_3d.shape[2]):
                output_array[tuple(point)] = 1.0

        return output_array


########################
# 3D to 2D projections #
########################
def _calculate_depth_projection(data_3d: np.ndarray, component_3d: np.ndarray = None, axis: int = 0):
    depth_projection = np.argmax(data_3d, axis=axis)
    max_projection = np.max(data_3d, axis=axis)
    max_axis_index = data_3d.shape[axis] - 1

    grayscale_depth_projection = np.where(
        max_projection > 0,
        np.round(255 * (1 - (depth_projection / max_axis_index))),
        0
    ).astype(np.uint8)

    depth_projects = dict()
    depth_projects["image"] = grayscale_depth_projection
    if component_3d is not None:
        components_depth_projection = np.zeros_like(grayscale_depth_projection)
        for i in range(grayscale_depth_projection.shape[0]):
            for j in range(grayscale_depth_projection.shape[1]):
                if grayscale_depth_projection[i, j] > 0:
                    if axis == 0:
                        components_depth_projection[i, j] = component_3d[depth_projection[i, j], i, j]
                    elif axis == 1:
                        components_depth_projection[i, j] = component_3d[i, depth_projection[i, j], j]
                    elif axis == 2:
                        components_depth_projection[i, j] = component_3d[i, j, depth_projection[i, j]]
                    else:
                        raise ValueError("Invalid axis")
        depth_projects["components"] = components_depth_projection

    return depth_projects

    # return (255 * (1 - (depth_projection / axis_size))).astype(int)


# TODO: Add support for data rotation
# TODO: nifty needs no data rotation, but matrix transpose since the order is (Z, Y, X) and not (X, Y, Z)
# TODO: also the first 3 rotations shouldn't be needed after transpose (need to find the correct one)
def project_3d_to_2d(data_3d: np.ndarray,
                     projection_options: dict[str, bool],
                     source_data_filepath=None,
                     component_3d: np.ndarray = None) -> Union[Dict, Dict[str, np.ndarray]]:
    projections = dict()

    rotated_data_3d = data_3d
    rotated_component_3d = component_3d

    if source_data_filepath is None:
        pass # No need for rotation

    # Medical data (nii.gz) has different axis order
    elif str(source_data_filepath).endswith(".nii.gz") is True:
        rotated_data_3d = np.rot90(rotated_data_3d, k=1, axes=(0, 2))  # For OpenCV compatibility
        if component_3d is not None:
            rotated_component_3d = np.rot90(rotated_component_3d, k=1, axes=(0, 2))

    # Other data formats
    else:
        rotated_data_3d = np.rot90(rotated_data_3d, k=1, axes=(0, 2))  # For OpenCV compatibility
        rotated_data_3d = np.rot90(rotated_data_3d, k=1, axes=(0, 1))
        if component_3d is not None:
            rotated_component_3d = np.rot90(rotated_component_3d, k=1, axes=(0, 2))
            rotated_component_3d = np.rot90(rotated_component_3d, k=1, axes=(0, 1))

    # Front projection (XZ plane)
    if projection_options.get("front", False) is True:
        flipped_data_3d = rotated_data_3d

        # Option 1
        # projections["front_image"] = np.max(data_3d, axis=2)

        # Option 2
        flipped_component_3d = rotated_component_3d
        if flipped_component_3d is not None:
           pass  # No need for rotation

        depth_projections = _calculate_depth_projection(
            data_3d=flipped_data_3d,
            component_3d=flipped_component_3d,
            axis=1
        )
        projections["front_image"] = depth_projections.get("image", None)
        projections["front_components"] = depth_projections.get("components", None)  # Optional

    # Back projection (XZ plane)
    if projection_options.get("back", False) is True:
        flipped_data_3d = rotated_data_3d
        flipped_data_3d = np.rot90(flipped_data_3d, k=2, axes=(1, 2))

        # Option 1
        # projections["back_image"] = np.max(flipped_data_3d, axis=2)

        # Option 2
        flipped_component_3d = rotated_component_3d
        if flipped_component_3d is not None:
            flipped_component_3d = np.rot90(flipped_component_3d, k=2, axes=(1, 2))

        depth_projections = _calculate_depth_projection(
            data_3d=flipped_data_3d,
            component_3d=flipped_component_3d,
            axis=1
        )
        projections["back_image"] = depth_projections.get("image", None)
        projections["back_components"] = depth_projections.get("components", None)  # Optional


    # Top projection (XY plane)
    if projection_options.get("top", False) is True:
        flipped_data_3d = rotated_data_3d
        flipped_data_3d = np.rot90(flipped_data_3d, k=1, axes=(0, 1))

        # Option 1
        # projections["top_image"] = np.max(data_3d, axis=1)

        # Option 2
        flipped_component_3d = rotated_component_3d
        if flipped_component_3d is not None:
            flipped_component_3d = np.rot90(flipped_component_3d, k=1, axes=(0, 1))

        depth_projections = _calculate_depth_projection(
            data_3d=flipped_data_3d,
            component_3d=flipped_component_3d,
            axis=1
        )
        projections["top_image"] = depth_projections.get("image", None)
        projections["top_components"] = depth_projections.get("components", None)  # Optional


    # Bottom projection (XY plane)
    if projection_options.get("bottom", False) is True:
        flipped_data_3d = rotated_data_3d
        flipped_data_3d = np.rot90(flipped_data_3d, k=1, axes=(1, 0))

        # Option 1
        # projections["bottom_image"] = np.max(flipped_data_3d, axis=1)

        # Option 2
        flipped_component_3d = rotated_component_3d
        if flipped_component_3d is not None:
            flipped_component_3d = np.rot90(flipped_component_3d, k=1, axes=(1, 0))

        depth_projections = _calculate_depth_projection(
            data_3d=flipped_data_3d,
            component_3d=flipped_component_3d,
            axis=1
        )
        projections["bottom_image"] = depth_projections.get("image", None)
        projections["bottom_components"] = depth_projections.get("components", None)  # Optional

    # Right projection (YZ plane)
    if projection_options.get("right", False) is True:
        flipped_data_3d = rotated_data_3d
        flipped_data_3d = np.rot90(flipped_data_3d, k=1, axes=(1, 2))

        # Option 1
        # projections["right_image"] = np.max(flipped_data_3d, axis=0)

        # Option 2
        flipped_component_3d = rotated_component_3d
        if flipped_component_3d is not None:
            flipped_component_3d = np.rot90(flipped_component_3d, k=1, axes=(1, 2))

        depth_projections = _calculate_depth_projection(
            data_3d=flipped_data_3d,
            component_3d=flipped_component_3d,
            axis=1
        )
        projections["right_image"] = depth_projections.get("image", None)
        projections["right_components"] = depth_projections.get("components", None)  # Optional

    # Left projection (YZ plane)
    if projection_options.get("left", False) is True:
        flipped_data_3d = rotated_data_3d
        flipped_data_3d = np.rot90(flipped_data_3d, k=1, axes=(2, 1))

        # Option 1
        # projections["left_image"] = np.max(data_3d, axis=0)

        # Option 2
        flipped_component_3d = rotated_component_3d
        if flipped_component_3d is not None:
            flipped_component_3d = np.rot90(flipped_component_3d, k=1, axes=(2, 1))

        depth_projections = _calculate_depth_projection(
            data_3d=flipped_data_3d,
            component_3d=flipped_component_3d,
            axis=1
        )
        projections["left_image"] = depth_projections.get("image", None)
        projections["left_components"] = depth_projections.get("components", None)  # Optional

    return projections


#############################
# 3D from 2D reconstruction #
#############################

# TODO: Add support for data rotation
def reverse_rotations(numpy_image: np.ndarray,
                      view_type: str,
                      source_data_filepath=None) -> np.ndarray:
    # Assumption: The source volume should be a cube!

    axis = 1  # Default axis for the 2D images
    max_axis_index = numpy_image.shape[0] - 1

    # Convert to 3D
    data_3d = np.zeros(shape=(numpy_image.shape[0], numpy_image.shape[0], numpy_image.shape[0]), dtype=np.uint8)
    for i in range(numpy_image.shape[0]):
        for j in range(numpy_image.shape[1]):
            gray_value = round(numpy_image[i, j])
            if gray_value > 0:
                rescale_gray_value = round(max_axis_index * (1 - (gray_value / 255)))

                if axis == 0:
                    data_3d[i, j, rescale_gray_value] = 1
                elif axis == 1:
                    data_3d[i, rescale_gray_value, j] = 1
                elif axis == 2:
                    data_3d[rescale_gray_value, i, j] = 1
                else:
                    raise ValueError("Invalid view type")

    # Reverse the rotations
    if view_type == "front":
        pass  # No need for rotation

    if view_type == "back":
        data_3d = np.rot90(data_3d, k=2, axes=(2, 1))

    if view_type == "top":
        data_3d = np.rot90(data_3d, k=1, axes=(1, 0))

    if view_type == "bottom":
        data_3d = np.rot90(data_3d, k=1, axes=(0, 1))

    if view_type == "right":
        data_3d = np.rot90(data_3d, k=1, axes=(2, 1))

    if view_type == "left":
        data_3d = np.rot90(data_3d, k=1, axes=(1, 2))

    # Reverse the initial rotations
    if source_data_filepath is None:
        pass
    # TODO: check how to use the angles correctly
    elif str(source_data_filepath).endswith(".nii.gz") is True:
        data_3d = np.rot90(data_3d, k=1, axes=(2, 0))
    else:
        data_3d = np.rot90(data_3d, k=1, axes=(1, 0))
        data_3d = np.rot90(data_3d, k=1, axes=(2, 0))

    return data_3d


# TODO: Do changes in all required places
def get_images_6_views(format_of_2d_images: str,
                       convert_to_3d: bool = False,
                       source_data_filepath=None) -> list:
    data_list = list()
    for image_view in IMAGES_6_VIEWS:
        image_path = format_of_2d_images.replace("<VIEW>", image_view)
        numpy_image = convert_data_file_to_numpy(data_filepath=image_path)
        # numpy_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if convert_to_3d is True:
            data_3d = reverse_rotations(
                numpy_image=numpy_image,
                view_type=image_view,
                source_data_filepath=source_data_filepath
            )
            data_list.append(data_3d)
        else:
            data_list.append(numpy_image)

    return data_list


# TODO: Do changes in all required places
def reconstruct_3d_from_2d(format_of_2d_images, source_data_filepath=None) -> np.ndarray:
    data_list = get_images_6_views(
        format_of_2d_images=format_of_2d_images,
        convert_to_3d=True,
        source_data_filepath=source_data_filepath
    )

    merged_data_3d = data_list[0]
    for i in range(1, len(data_list)):
        merged_data_3d = np.logical_or(merged_data_3d, data_list[i])
    merged_data_3d = merged_data_3d.astype(np.float32)
    apply_threshold(merged_data_3d, threshold=0.5, keep_values=False)

    # save_name = format_of_2d_images.replace("<VIEW>", "result")
    # convert_numpy_to_nii_gz(merged_data_3d, save_name=save_name)

    return merged_data_3d


##############################
# Interactive Visualizations #
##############################

# Interactive Plot 3D
def _interactive_plot_3d(data_3d: np.ndarray, version: int = 1, **kwargs):
    """
    Interactive 3D plot using Plotly or Matplotlib
    :param data_3d:
    :param version:
    :param kwargs: set_aspect_ratios, downsample_factor
    :return:
    """
    if version == 1:
        threshold = 0.5 * np.max(data_3d)
        x, y, z = np.where(data_3d > threshold)

        # Create the Plotly 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=2,
                color=data_3d[x, y, z],
                colorscale='Viridis',
                opacity=0.6
            )
        )])

        # aspect_ratios = data_3d.shape  # Lengths of each dimension
        fig.update_layout(scene=dict(
            xaxis_title='X-axis',
            yaxis_title='Y-axis',
            zaxis_title='Z-axis',
            # aspectmode="manual",  # Use manual aspect ratios
            # aspectratio=dict(
            #     x=aspect_ratios[0] / max(aspect_ratios),
            #     y=aspect_ratios[1] / max(aspect_ratios),
            #     z=aspect_ratios[2] / max(aspect_ratios),
            # )
        ), title="3D Volume Visualization")

        fig.show()

    elif version == 2:
        # Downsample the images
        downsample_factor = kwargs.get('downsample_factor', 1)
        data_downsampled = data_3d[::downsample_factor, ::downsample_factor, ::downsample_factor]

        # Get the indices of non-zero values in the downsampled array
        nonzero_indices = np.where(data_downsampled != 0)

        # Create a figure and a 3D axis
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Get the permutation
        if data_3d.max() > 1:
            color_mode = True
        else:
            color_mode = False

        array_value = [
            nonzero_indices[0],
            nonzero_indices[1],
            nonzero_indices[2]
        ]
        if color_mode is True:
            color_value = data_downsampled[
                nonzero_indices[0],
                nonzero_indices[1],
                nonzero_indices[2]
            ]
            color_value = color.label2rgb(label=color_value)
            ax.bar3d(*array_value, 1, 1, 1, color=color_value)
        else:
            ax.bar3d(*array_value, 1, 1, 1, color='b')

        # ax.bar3d(nonzero_indices[0], nonzero_indices[1], nonzero_indices[2], 1, 1, 1, color='b')

        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Set aspect ratios
        set_aspect_ratios = kwargs.get('set_aspect_ratios', False)
        if set_aspect_ratios is True:
            aspect_ratios = np.array(
                [data_3d.shape[0], data_3d.shape[1], data_3d.shape[2]])  # Use the actual shape of the volume
            ax.set_box_aspect(aspect_ratios)

        # Display the plot
        plt.title('3d plot')
        plt.show()
    else:
        raise ValueError("Invalid version number. Please use either 1 or 2.")

# Interactive Plot 2D
def _interactive_plot_2d(data_2d: np.ndarray, apply_label2rgb: bool = False):
    """
    Interactive 2D plot using Matplotlib
    :param data_2d:
    :param apply_label2rgb:
    :return:
    """
    if apply_label2rgb:
        colored_data = color.label2rgb(label=data_2d)
    else:
        colored_data = data_2d

    if colored_data.shape == 3:
        plt.imshow(colored_data)
    else:
        plt.imshow(colored_data, cmap='gray')
    plt.show()
