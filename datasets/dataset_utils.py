import pathlib
import numpy as np
import torch
import cv2

# For .nii.gz
import nibabel as nib

# For .ply
import trimesh
import open3d as o3d


IMAGES_6_VIEWS = ['top', 'bottom', 'front', 'back', 'left', 'right']


def get_data_file_stem(data_filepath) -> str:
    data_filepath = str(data_filepath)
    if data_filepath.endswith(".nii.gz"):
        data_filepath_stem = pathlib.Path(data_filepath.replace(".nii.gz", "")).name
    elif data_filepath.endswith(".ply"):
        data_filepath_stem =  pathlib.Path(data_filepath.replace(".ply", "")).name
    else:
        raise ValueError("Invalid data format")

    return data_filepath_stem


def convert_data_file_to_numpy(data_filepath) -> np.ndarray:
    data_filepath = str(data_filepath)
    if data_filepath.endswith(".nii.gz"):
        numpy_data = _convert_nii_gz_to_numpy(data_filepath=data_filepath)
    elif data_filepath.endswith(".ply"):
        numpy_data = _convert_ply_to_numpy(data_filepath=data_filepath)
    elif data_filepath.endswith(".obj"):
        numpy_data = _convert_obj_to_numpy(data_filepath=data_filepath)
    elif data_filepath.endswith(".pcd"):
        numpy_data = _convert_pcd_to_numpy(data_filepath=data_filepath)
    elif data_filepath.endswith(".npy"):
        numpy_data = _convert_npy_to_numpy(data_filepath=data_filepath)
    else:
        raise ValueError("Invalid data format")

    return numpy_data


def convert_numpy_to_data_file(numpy_data: np.ndarray, source_data_filepath, save_filename=None):
    source_data_filepath = str(source_data_filepath)
    if source_data_filepath.endswith(".nii.gz"):
        _convert_numpy_to_nii_gz(numpy_data=numpy_data, source_data_filepath=source_data_filepath,
                                 save_filename=save_filename)
    elif source_data_filepath.endswith(".ply"):
        _convert_numpy_to_ply(numpy_data=numpy_data, source_data_filepath=source_data_filepath,
                              save_filename=save_filename)
    elif source_data_filepath.endswith(".obj"):
        _convert_numpy_to_obj(numpy_data=numpy_data, source_data_filepath=source_data_filepath,
                              save_filename=save_filename)
    elif source_data_filepath.endswith(".pcd"):
        _convert_numpy_to_pcd(numpy_data=numpy_data, source_data_filepath=source_data_filepath,
                              save_filename=save_filename)
    elif source_data_filepath.endswith(".npy"):
        _convert_numpy_to_npy(numpy_data=numpy_data, source_data_filepath=source_data_filepath,
                              save_filename=save_filename)
    else:
        raise ValueError("Invalid data format")


#######################################
# nii.gz to numpy and numpy to nii.gz #
#######################################

# TODO: return or apply the affine transformation to the numpy data for the save later
def _convert_nii_gz_to_numpy(data_filepath: str) -> np.ndarray:
    nifti_data = nib.load(data_filepath)
    numpy_data = nifti_data.get_fdata()
    return numpy_data


def _convert_numpy_to_nii_gz(numpy_data: np.ndarray, source_data_filepath: str = None,
                             save_filename=None) -> nib.Nifti1Image:
    if source_data_filepath is not None:
        nifti_data = nib.load(source_data_filepath)
        new_nifti_data = nib.Nifti1Image(numpy_data, affine=nifti_data.affine, header=nifti_data.header)
        if "int" in str(numpy_data.dtype):  # Keep integer type for components data
            new_nifti_data.header.set_data_dtype(np.int16) # Assumption: Max value will be less than 32767
    else:
        new_nifti_data = nib.Nifti1Image(numpy_data, affine=np.eye(4))

    if save_filename is not None and len(save_filename) > 0:
        save_filename = str(save_filename)
        if not save_filename.endswith(".nii.gz"):
            save_filename = f"{save_filename}.nii.gz"
        nib.save(img=new_nifti_data, filename=save_filename)  # Save to NII.GZ
    return new_nifti_data


def save_nii_gz_in_identity_affine(numpy_data=None, data_filepath=None, save_filename=None):
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
        if not save_filename.endswith(".nii.gz"):
            save_filename = f"{save_filename}.nii.gz"
        nib.save(img=new_nifti_data, filename=save_filename)  # Save to NII.GZ
    return new_nifti_data


#################################
# ply to numpy and numpy to ply #
#################################

# TODO: Check how to make Abstract converter from supported file formats

def _convert_ply_to_numpy(data_filepath) -> np.ndarray:
    voxel_size = 0.05  # Define voxel size (the size of each grid cell)

    # Mesh PLY
    if data_filepath.endswith("mesh.ply"):
        mesh = trimesh.load(data_filepath)
        voxelized = mesh.voxelized(pitch=voxel_size)  # Pitch = voxel size

        numpy_data = voxelized.matrix.astype(np.uint8)
    # Point Cloud PLY
    elif data_filepath.endswith("pcd.ply"):
        pcd = o3d.io.read_point_cloud(data_filepath)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(input=pcd, voxel_size=voxel_size)  # Voxelize pcd
        voxels = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])  # Get voxel centers
        max_bounds = np.max(voxels, axis=0)
        min_bounds = np.min(voxels, axis=0)
        grid_shape = max_bounds - min_bounds + 1  # Determine grid dimensions

        numpy_data = np.zeros(shape=grid_shape, dtype=np.uint8)
        for voxel in voxels:
            grid_index = voxel - min_bounds  # Shift indices to start at 0
            numpy_data[tuple(grid_index)] = 1  # 1 = occupied, 0 = empty
    else:
        raise ValueError("Invalid data format")

    return numpy_data


def _convert_numpy_to_ply(numpy_data: np.ndarray, source_data_filepath=None, save_filename=None):
    voxel_size = 0.05  # Define voxel size (the size of each grid cell)

    # Mesh PLY
    if source_data_filepath.endswith("mesh.ply"):
        occupied_indices = np.argwhere(numpy_data == 1)  # Find occupied voxels (indices where numpy_data == 1)
        centers = occupied_indices * voxel_size  # Convert indices to real-world coordinates (Scale by voxel size)

        # Create cube meshes for each occupied voxel
        cubes = []
        for center in centers:
            # Create a cube for each voxel
            cube = trimesh.creation.box(
                extents=[voxel_size] * 3,
                transform=trimesh.transformations.translation_matrix(center)
            )
            cubes.append(cube)
        mesh = trimesh.util.concatenate(cubes) # Combine all cubes into a single mesh

        if save_filename is not None and len(save_filename) > 0:
            save_filename = str(save_filename)
            if not save_filename.endswith("mesh.ply"):
                save_filename = f"{save_filename}_mesh.ply"
            mesh.export(file_obj=save_filename)  # Save to PLY

        new_ply_data = mesh

    # Point Cloud PLY
    elif source_data_filepath.endswith("pcd.ply"):
        occupied_indices = np.argwhere(numpy_data == 1)  # Find occupied voxels (indices where numpy_data == 1)
        points = occupied_indices * voxel_size  # Convert indices to real-world coordinates (Scale by voxel size)

        pcd = o3d.geometry.PointCloud() # Create Open3D PointCloud
        pcd.points = o3d.utility.Vector3dVector(points)

        if save_filename is not None and len(save_filename) > 0:
            save_filename = str(save_filename)
            if not save_filename.endswith("pcd.ply"):
                save_filename = f"{save_filename}_pcd.ply"
            o3d.io.write_point_cloud(filename=save_filename, pointcloud=pcd)  # Save to PLY

        new_ply_data = pcd
    else:
        raise ValueError("Invalid data format")

    return new_ply_data


#################################
# obj to numpy and numpy to obj #
#################################
def _convert_obj_to_numpy(data_filepath) -> np.ndarray:
    voxel_size = 2.0  # Define voxel size (the size of each grid cell)

    mesh = trimesh.load(data_filepath)
    voxelized = mesh.voxelized(pitch=voxel_size)  # Pitch = voxel size

    numpy_data = voxelized.matrix.astype(np.uint8)
    return numpy_data


def _convert_numpy_to_obj(numpy_data: np.ndarray, source_data_filepath=None, save_filename=None) -> trimesh.Trimesh:
    voxel_size = 2.0  # Define voxel size (the size of each grid cell)

    occupied_indices = np.argwhere(numpy_data == 1)  # Find occupied voxels (indices where numpy_data == 1)
    centers = occupied_indices * voxel_size  # Convert indices to real-world coordinates (Scale by voxel size)

    # Create cube meshes for each occupied voxel
    cubes = []
    for center in centers:
        # Create a cube for each voxel
        cube = trimesh.creation.box(
            extents=[voxel_size] * 3,
            transform=trimesh.transformations.translation_matrix(center)
        )
        cubes.append(cube)
    mesh = trimesh.util.concatenate(cubes)  # Combine all cubes into a single mesh

    if save_filename is not None and len(save_filename) > 0:
        save_filename = str(save_filename)
        if not save_filename.endswith("obj"):
            save_filename = f"{save_filename}.obj"
        mesh.export(file_obj=save_filename)  # Save to OBJ

    new_obj_data = mesh
    return new_obj_data


#################################
# pcd to numpy and numpy to pcd #
#################################
def _convert_pcd_to_numpy(data_filepath) -> np.ndarray:
    voxel_size = 2.0  # Define voxel size (the size of each grid cell)

    pcd = o3d.io.read_point_cloud(data_filepath)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(input=pcd, voxel_size=voxel_size)  # Voxelize pcd
    voxels = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])  # Get voxel centers
    max_bounds = np.max(voxels, axis=0)
    min_bounds = np.min(voxels, axis=0)
    grid_shape = max_bounds - min_bounds + 1  # Determine grid dimensions

    numpy_data = np.zeros(shape=grid_shape, dtype=np.uint8)
    for voxel in voxels:
        grid_index = voxel - min_bounds  # Shift indices to start at 0
        numpy_data[tuple(grid_index)] = 1  # 1 = occupied, 0 = empty

    return numpy_data


def _convert_numpy_to_pcd(numpy_data: np.ndarray, source_data_filepath=None, save_filename=None) -> o3d.geometry.PointCloud:
    voxel_size = 2.0  # Define voxel size (the size of each grid cell)

    occupied_indices = np.argwhere(numpy_data == 1)  # Find occupied voxels (indices where numpy_data == 1)
    points = occupied_indices * voxel_size  # Convert indices to real-world coordinates (Scale by voxel size)

    pcd = o3d.geometry.PointCloud()  # Create Open3D PointCloud
    pcd.points = o3d.utility.Vector3dVector(points)

    if save_filename is not None and len(save_filename) > 0:
        save_filename = str(save_filename)
        if not save_filename.endswith(".pcd"):
            save_filename = f"{save_filename}.pcd"
        o3d.io.write_point_cloud(filename=save_filename, pointcloud=pcd)  # Save to PCD

    new_pcd_data = pcd
    return new_pcd_data


#################################
# npy to numpy and numpy to npy #
#################################
def _convert_npy_to_numpy(data_filepath) -> np.ndarray:
    numpy_data = np.load(file=data_filepath)
    return numpy_data


def _convert_numpy_to_npy(numpy_data: np.ndarray, source_data_filepath=None, save_filename=None) -> np.ndarray:
    if save_filename is not None and len(save_filename) > 0:
        save_filename = str(save_filename)
        if not save_filename.endswith(".npy"):
            save_filename = f"{save_filename}.npy"
        np.save(file=save_filename, arr=numpy_data)  # Save to NPY
    return numpy_data


################
# Thresholding #
################
def apply_threshold(tensor: torch.Tensor, threshold: float):
    tensor[tensor >= threshold] = 1.0
    tensor[tensor < threshold] = 0.0


########################
# 3D to 2D projections #
########################
def _calculate_depth_projection(data_3d, component_3d=None, axis=0):
    depth_projection = np.argmax(data_3d, axis=axis)
    max_projection = np.max(data_3d, axis=axis)
    axis_size = data_3d.shape[axis]

    grayscale_depth_projection = np.where(
        max_projection > 0,
        (255 * (1 - (depth_projection / axis_size))).astype(int),
        0
    )

    if component_3d is None:
        return grayscale_depth_projection
    else:
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

        return grayscale_depth_projection, components_depth_projection

    # return (255 * (1 - (depth_projection / axis_size))).astype(int)


def project_3d_to_2d(data_3d: np.ndarray,
                     component_3d: np.ndarray = None,
                     front=False,
                     back=False,
                     top=False,
                     bottom=False,
                     left=False,
                     right=False):
    projections = dict()

    rotated_data_3d = data_3d
    rotated_data_3d = np.rot90(rotated_data_3d, k=1, axes=(0, 2))
    rotated_data_3d = np.rot90(rotated_data_3d, k=1, axes=(1, 2))
    rotated_data_3d = np.flip(rotated_data_3d, axis=1)

    if component_3d is not None:
        rotated_component_3d = component_3d
        rotated_component_3d = np.rot90(rotated_component_3d, k=1, axes=(0, 2))
        rotated_component_3d = np.rot90(rotated_component_3d, k=1, axes=(1, 2))
        rotated_component_3d = np.flip(rotated_component_3d, axis=1)
    else:
        rotated_component_3d = None

    # Front projection (XY plane)
    if front is True:
        flipped_data_3d = rotated_data_3d

        # Option 1
        # projections["front_image"] = np.max(data_3d, axis=2)

        # Option 2
        if rotated_component_3d is None:
            projections["front_image"] = _calculate_depth_projection(data_3d=flipped_data_3d, axis=2)
        else:
            flipped_component_3d = rotated_component_3d

            projections["front_image"], projections["front_components"] = _calculate_depth_projection(
                data_3d=flipped_data_3d,
                component_3d=flipped_component_3d,
                axis=2
            )

    # Back projection (XY plane)
    if back is True:
        flipped_data_3d = rotated_data_3d
        flipped_data_3d = np.rot90(flipped_data_3d, k=2, axes=(1, 2))

        # Option 1
        # projections["back_image"] = np.max(flipped_data_3d, axis=2)

        # Option 2
        if rotated_component_3d is None:
            projections["back_image"] = _calculate_depth_projection(data_3d=flipped_data_3d, axis=2)
        else:
            flipped_component_3d = rotated_component_3d
            flipped_component_3d = np.rot90(flipped_component_3d, k=2, axes=(1, 2))

            projections["back_image"], projections["back_components"] = _calculate_depth_projection(
                data_3d=flipped_data_3d,
                component_3d=flipped_component_3d,
                axis=2
            )

    # Top projection (XZ plane)
    if top is True:
        flipped_data_3d = rotated_data_3d
        flipped_data_3d = np.rot90(flipped_data_3d, k=1, axes=(1, 2))

        # Option 1
        # projections["top_image"] = np.max(data_3d, axis=1)

        # Option 2
        if rotated_component_3d is None:
            projections["top_image"] = _calculate_depth_projection(data_3d=flipped_data_3d, axis=0)
        else:
            flipped_component_3d = rotated_component_3d
            flipped_component_3d = np.rot90(flipped_component_3d, k=1, axes=(1, 2))

            projections["top_image"], projections["top_components"] = _calculate_depth_projection(
                data_3d=flipped_data_3d,
                component_3d=flipped_component_3d,
                axis=0
            )

    # Bottom projection (XZ plane)
    if bottom is True:
        flipped_data_3d = rotated_data_3d
        flipped_data_3d = np.rot90(flipped_data_3d, k=1, axes=(1, 2))
        flipped_data_3d = np.rot90(flipped_data_3d, k=2, axes=(0, 1))

        # Option 1
        # projections["bottom_image"] = np.max(flipped_data_3d, axis=1)

        # Option 2
        if rotated_component_3d is None:
            projections["bottom_image"] = _calculate_depth_projection(data_3d=flipped_data_3d, axis=0)
        else:
            flipped_component_3d = rotated_component_3d
            flipped_component_3d = np.rot90(flipped_component_3d, k=1, axes=(1, 2))
            flipped_component_3d = np.rot90(flipped_component_3d, k=2, axes=(0, 1))

            projections["bottom_image"], projections["bottom_components"] = _calculate_depth_projection(
                data_3d=flipped_data_3d,
                component_3d=flipped_component_3d,
                axis=0
            )

    # Right projection (YZ plane)
    if right is True:
        flipped_data_3d = rotated_data_3d
        flipped_data_3d = np.flip(flipped_data_3d, axis=1)

        # Option 1
        # projections["right_image"] = np.max(flipped_data_3d, axis=0)

        # Option 2
        if rotated_component_3d is None:
            projections["right_image"] = _calculate_depth_projection(data_3d=flipped_data_3d, axis=1)
        else:
            flipped_component_3d = rotated_component_3d
            flipped_component_3d = np.flip(flipped_component_3d, axis=1)

            projections["right_image"], projections["right_components"] = _calculate_depth_projection(
                data_3d=flipped_data_3d,
                component_3d=flipped_component_3d,
                axis=1
            )

    # Left projection (YZ plane)
    if left is True:
        flipped_data_3d = rotated_data_3d
        flipped_data_3d = np.flip(flipped_data_3d, axis=1)
        flipped_data_3d = np.rot90(flipped_data_3d, k=2, axes=(1, 2))

        # Option 1
        # projections["left_image"] = np.max(data_3d, axis=0)

        # Option 2
        if rotated_component_3d is None:
            projections["left_image"] = _calculate_depth_projection(data_3d=flipped_data_3d, axis=1)
        else:
            flipped_component_3d = rotated_component_3d
            flipped_component_3d = np.flip(flipped_component_3d, axis=1)
            flipped_component_3d = np.rot90(flipped_component_3d, k=2, axes=(1, 2))

            projections["left_image"], projections["left_components"] = _calculate_depth_projection(
                data_3d=flipped_data_3d,
                component_3d=flipped_component_3d,
                axis=1
            )

    return projections


#############################
# 3D from 2D reconstruction #
#############################
def reverse_rotations(numpy_image: np.ndarray, view_type: str) -> np.ndarray:
    # Convert to 3D
    data_3d = np.zeros(shape=(numpy_image.shape[0], numpy_image.shape[0], numpy_image.shape[0]), dtype=np.uint8)
    for i in range(numpy_image.shape[0]):
        for j in range(numpy_image.shape[1]):
            gray_value = int(numpy_image[i, j])
            if gray_value > 0:
                rescale_gray_value = int(numpy_image.shape[0] * (1 - (gray_value / 255)))

                if view_type in ["front", "back"]:
                    data_3d[i, j, rescale_gray_value] = 1
                elif view_type in ["top", "bottom"]:
                    data_3d[rescale_gray_value, i, j] = 1
                elif view_type in ["right", "left"]:
                    data_3d[i, rescale_gray_value, j] = 1
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


def get_images_6_views(format_of_2d_images: str, convert_to_3d: bool = False) -> list:
    data_list = list()
    for image_view in IMAGES_6_VIEWS:
        image_path = format_of_2d_images.replace("<VIEW>", image_view)
        numpy_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if convert_to_3d is True:
            data_3d = reverse_rotations(numpy_image, image_view)
            data_list.append(data_3d)
        else:
            data_list.append(numpy_image)

    return data_list


def reconstruct_3d_from_2d(format_of_2d_images) -> np.ndarray:
    data_list = get_images_6_views(format_of_2d_images=format_of_2d_images, convert_to_3d=True)

    merged_data_3d = data_list[0]
    for i in range(1, len(data_list)):
        merged_data_3d = np.logical_or(merged_data_3d, data_list[i])
    merged_data_3d = merged_data_3d.astype(np.float32)

    # save_name = format_of_2d_images.replace("<VIEW>", "result")
    # convert_numpy_to_nii_gz(merged_data_3d, save_name=save_name)

    return merged_data_3d
