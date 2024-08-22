import numpy as np
import nibabel as nib
import cv2


#########
# Utils #
#########
def convert_nii_to_numpy(data_file):
    ct_img = nib.load(data_file)
    ct_numpy = ct_img.get_fdata()
    return ct_numpy


def convert_to_3d_points(data_3d, normalize=True):
    geo = np.where(data_3d > 0)
    geo = np.array(geo)
    geo = geo.T

    if normalize:
        geo = geo.astype(np.float32)
        for i in range(3):
            max_val = geo.T[:][i].max()
            min_val = geo.T[:][i].min()
            geo.T[:][i] = 2 * ((geo.T[:][i] - min_val) / (max_val - min_val)) - 1  # Normalize to [-1, 1]
            # geo.T[:][i] = geo.T[:][i] / max_val)  # Normalize to [0, 1]

        # geo.T[:][0] = geo.T[:][0] / (geo.T[:][0].max())
        # geo.T[:][1] = geo.T[:][1] / (geo.T[:][1].max())
        # geo.T[:][2] = geo.T[:][2] / (geo.T[:][2].max())
    return geo


def project_points(ct_numpy):
    geo = convert_to_3d_points(ct_numpy, True)

    # Define intrinsic camera parameters
    focal_length = 500
    image_width = 640
    image_height = 640
    intrinsic_matrix = np.array([
        [focal_length, 0, image_width / 2],
        [0, focal_length, image_height / 2],
        [0, 0, 1]
    ])

    # Define extrinsic camera parameters
    rvec = np.array([0, 0, 0], dtype=np.float32)
    tvec = np.array([0, 0, 3], dtype=np.float32)

    # Generate 3D points on a paraboloid
    # u_range = np.linspace(-1, 1, num=20)
    # v_range = np.linspace(-1, 1, num=20)
    # u, v = np.meshgrid(u_range, v_range)
    # x = u
    # y = v
    # z = u ** 2 + v ** 2

    # points_3d = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    # Project 3D points onto 2D plane
    points_2d, _ = cv2.projectPoints(objectPoints=geo,
                                     rvec=rvec,
                                     tvec=tvec.T,
                                     cameraMatrix=intrinsic_matrix,
                                     distCoeffs=None)

    # Plot 2D points
    img = np.zeros(shape=(image_height, image_width), dtype=np.uint8)
    for point in points_2d.astype(int):
        img = cv2.circle(img=img, center=tuple(point[0]), radius=1, color=255, thickness=-1)

    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    # Load the data
    data_file = r"..\skel_np\PA000005.nii.gz"
    ct_numpy = convert_nii_to_numpy(data_file)
    project_points(ct_numpy)


if __name__ == "__main__":
    # TODO:
    # 1. Make sure "skel_np" folder is present in the root directory
    main()
