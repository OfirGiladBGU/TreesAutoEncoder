import open3d as o3d


if __name__ == '__main__':
    # File paths (update with your file paths if needed)
    data_filepath = r".\46_071_output.pcd"

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)
    # Call only after creating visualizer window.
    vis.get_render_option().background_color = [0, 0, 0]

    # Visualize the files
    if data_filepath.endswith(".obj"):
        mesh = o3d.io.read_triangle_mesh(data_filepath)  # For mesh files
        # o3d.visualization.draw_geometries([mesh], window_name="Mesh Viewer")
        vis.add_geometry(mesh)
    else:
        pcd = o3d.io.read_point_cloud(data_filepath) # For point cloud files
        # o3d.visualization.draw_geometries([pcd], window_name="Point Cloud Viewer")
        vis.add_geometry(pcd)

    vis.run()
