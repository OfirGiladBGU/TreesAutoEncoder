from dataset_utils import save_nii_gz_in_identity_affine, _convert_numpy_to_nii_gz
#
# data_filepath=r"C:\Users\ofirg\PycharmProjects\TreesAutoEncoder\data\parse2022\preds_components\PA000005_vessel.nii.gz"
# save_nii_gz_in_identity_affine(data_filepath=data_filepath, save_filename="test")

import open3d as o3d
import numpy as np

# File paths (update with your file paths if needed)
file_1 = r"..\data\PyPipes\label\0_mesh.ply"
file_2 = r"..\data\PyPipes\label\0_pcd.ply"

# Load the PLY files
# mesh = o3d.io.read_triangle_mesh(file_1)  # For mesh files
pcd = o3d.io.read_point_cloud(file_2)

# For point cloud files
# Define voxel size (the size of each grid cell)
voxel_size = 0.05  # Adjust based on your data's scale

# Voxelize the point cloud
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)

# Convert to a 3D NumPy array
# Get voxel centers
voxels = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])

# Optional: Create a dense 3D grid (binary representation)
# Determine grid dimensions
min_bounds = np.min(voxels, axis=0)
max_bounds = np.max(voxels, axis=0)
grid_shape = max_bounds - min_bounds + 1

# Create a binary voxel grid (1 = occupied, 0 = empty)
voxel_array = np.zeros(grid_shape, dtype=np.uint8)
for voxel in voxels:
    grid_index = voxel - min_bounds  # Shift indices to start at 0
    voxel_array[tuple(grid_index)] = 1

_convert_numpy_to_nii_gz(voxel_array, save_filename="test_pcd")

######################################################

import trimesh

# Load the mesh
mesh = trimesh.load(file_1)

# Voxelize the mesh
voxelized = mesh.voxelized(pitch=0.05)  # Pitch = voxel size

# Get dense voxel representation as a 3D NumPy array
voxel_array = voxelized.matrix

print("3D Voxel Grid Shape:", voxel_array.shape)

voxel_array = voxel_array.astype(np.uint8)
_convert_numpy_to_nii_gz(voxel_array, save_filename="test_mesh")



# Print information about the loaded files
# print("Mesh:")
# print(mesh)
# print("Point Cloud:")
# print(pcd)

# Visualize the files
# o3d.visualization.draw_geometries([mesh], window_name="Mesh Viewer")
# o3d.visualization.draw_geometries([pcd], window_name="Point Cloud Viewer")