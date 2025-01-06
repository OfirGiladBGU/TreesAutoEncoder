import numpy as np
import open3d as o3d
import trimesh

# Create a simple mesh for demonstration
mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
mesh.compute_vertex_normals()


# Convert Open3D mesh to trimesh format
def convert_mesh_to_trimesh(mesh):
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    return trimesh.Trimesh(vertices=vertices, faces=faces)


# Convert Open3D mesh to trimesh
trimesh_mesh = convert_mesh_to_trimesh(mesh)

# Define the voxel size
voxel_size = 0.05  # Adjust voxel size as needed


def mesh_to_voxel_grid(trimesh_mesh, voxel_size):
    # Use trimesh's voxelize function
    voxel_grid = trimesh.voxel.creation.voxelize(trimesh_mesh, voxel_size)

    # Convert voxel grid to a numpy array
    voxel_matrix = voxel_grid.matrix.astype(bool)

    return voxel_matrix


def voxel_grid_to_point_cloud(voxel_grid, voxel_size):
    # Get voxel indices
    voxel_indices = np.argwhere(voxel_grid)

    # Convert voxel indices to 3D coordinates (centers of voxels)
    voxel_centers = voxel_indices * voxel_size + voxel_size / 2

    # Create a point cloud from voxel centers
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(voxel_centers)

    return point_cloud


# Convert mesh to voxel grid
voxel_grid = mesh_to_voxel_grid(trimesh_mesh, voxel_size)

# Convert voxel grid to Open3D Point Cloud
point_cloud = voxel_grid_to_point_cloud(voxel_grid, voxel_size)

# Visualize the point cloud using Open3D
o3d.visualization.draw_geometries([point_cloud])
