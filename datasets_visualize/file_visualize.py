import open3d as o3d
import os


def visualize():
    vis = o3d.visualization.Visualizer()
    window_name = f"Open3D - {os.path.basename(data_filepath)}"
    vis.create_window(visible=True, window_name=window_name)

    # Call only after creating visualizer window.
    vis.get_render_option().background_color = [0, 0, 0]

    # Visualize the files
    if data_filepath.endswith(".obj") or data_filepath.endswith("_mesh.ply"):
        mesh = o3d.io.read_triangle_mesh(data_filepath)  # For mesh files
        # o3d.visualization.draw_geometries([mesh], window_name=window_name)
        vis.add_geometry(mesh)
    elif data_filepath.endswith(".pcd") or data_filepath.endswith("_pcd.ply"):
        pcd = o3d.io.read_point_cloud(data_filepath)  # For point cloud files
        # o3d.visualization.draw_geometries([pcd], window_name=window_name)
        vis.add_geometry(pcd)
    else:
        raise ValueError("Not Supported")

    vis.run()


if __name__ == '__main__':
    # File paths (update with your file paths if needed)
    data_filepath = r".\output.pcd"

    visualize()
