import open3d as o3d
import numpy as np

def read_pts_file(pts_path):
    points = []
    colors = []

    with open(pts_path, 'r') as file:
        for line in file:
            tokens = line.strip().split()
            if len(tokens) >= 3:
                x, y, z = map(float, tokens[:3])
                points.append([x, y, z])

                # Enable colors
                # if len(tokens) >= 6:
                #     r, g, b = map(float, tokens[3:6])
                #     colors.append([r / 255.0, g / 255.0, b / 255.0])

    return np.array(points), np.array(colors) if colors else None

def convert_pts_to_pcd(pts_path, pcd_path):
    points, colors = read_pts_file(pts_path)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if colors is not None and len(colors) == len(points):
        pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(pcd_path, pcd)
    print(f"Saved PCD file to: {pcd_path}")

# Example usage
pts_file = "example.pts"
pcd_file = "output.pcd"
convert_pts_to_pcd(pts_file, pcd_file)
