import numpy as np
import nibabel as nib
import cv2
import random
import os
import pandas as pd


#########
# Utils #
#########
def convert_nii_to_numpy(data_file):
    ct_img = nib.load(data_file)
    ct_numpy = ct_img.get_fdata()
    return ct_numpy


def project_3d_to_2d(data_3d, front=False, front_cut=-1, up=False, up_cut=-1, left=False, left_cut=-1):
    projections = dict()

    # Front projection (XY plane)
    if front:
        if front_cut != -1:
            data_3d = data_3d[:, :, 0:front_cut]
        projections["front_image"] = np.max(data_3d, axis=2)

    # Up projection (XZ plane)
    if up:
        if up_cut != -1:
            data_3d = data_3d[:, 0:up_cut, :]
        projections["up_image"] = np.max(data_3d, axis=1)

    # Left projection (YZ plane)
    if left:
        if left_cut != -1:
            data_3d = data_3d[0:left_cut, :, :]
        projections["left_image"] = np.max(data_3d, axis=0)

    return projections


def crop_black_area(image):
    # Convert to Threshold image
    image = image.astype(dtype=np.uint8)
    image[image > 0] = 255

    # Find contours
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables for the bounding rectangle
    min_x, min_y, max_x, max_y = float('inf'), float('inf'), 0, 0

    # Iterate through contours and find the bounding rectangle
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)

    # Crop the image using the calculated bounding rectangle
    cropped_image = image[min_y:max_y, min_x:max_x]

    # Print the coordinates of the bounding rectangle
    print(f"Top-left: ({min_x}, {min_y})")
    print(f"Bottom-right: ({max_x}, {max_y})")

    crop_dim = (min_x, min_y, max_x, max_y)
    return cropped_image, crop_dim


def crop_3d(data_3d):
    projections = project_3d_to_2d(data_3d, front=True, up=True)
    front_image, front_dim = crop_black_area(projections["front_image"])
    up_image, up_dim = crop_black_area(projections["up_image"])
    x_min, y_min, x_max, y_max = front_dim
    z_min, _, z_max, _ = up_dim
    cropped_data_3d = data_3d[y_min:y_max, x_min:x_max, z_min:z_max]
    return cropped_data_3d


def crop_mini_cubes(cropped_data_3d, size=(28, 28, 28)):
    step = 14
    mini_cubes = []
    for i in range(0, cropped_data_3d.shape[0], step):
        for j in range(0, cropped_data_3d.shape[1], step):
            for k in range(0, cropped_data_3d.shape[2], step):
                # print(i, j, k)
                if (
                    i + size[0] > cropped_data_3d.shape[0] or
                    j + size[1] > cropped_data_3d.shape[1] or
                    k + size[2] > cropped_data_3d.shape[2]
                ):
                    continue
                mini_cube = cropped_data_3d[i:i+size[0], j:j+size[1], k:k+size[2]]
                mini_cubes.append(mini_cube)

    return mini_cubes


####################
# Original Dataset #
####################
def create_dataset_original_images():
    folder_path = "../skel_np"
    org_folder = "./mini_cropped_images"

    white_points_upper_threshold = 28 * 28 * 0.5
    white_points_lower_threshold = 10

    os.makedirs(org_folder, exist_ok=True)
    data_filepaths = os.listdir(folder_path)
    for data_filepath in data_filepaths:
        output_idx = data_filepath.split(".")[0]
        data_filepath = os.path.join(folder_path, data_filepath)
        ct_numpy = convert_nii_to_numpy(data_file=data_filepath)

        cropped_data_3d = crop_3d(ct_numpy)
        cropped_data_3d[cropped_data_3d > 0] = 255
        mini_cubes = crop_mini_cubes(cropped_data_3d)
        print("Total Mini Cubes:", len(mini_cubes))

        for mini_box_id, mini_cube in enumerate(mini_cubes):
            projections = project_3d_to_2d(mini_cube, front=True, up=True, left=True)
            front_image = projections["front_image"]
            up_image = projections["up_image"]
            left_image = projections["left_image"]

            if white_points_upper_threshold > np.count_nonzero(front_image) > white_points_lower_threshold:
                cv2.imwrite(f"{org_folder}/front_{output_idx}_{mini_box_id}.png", front_image)

            if white_points_upper_threshold > np.count_nonzero(up_image) > white_points_lower_threshold:
                cv2.imwrite(f"{org_folder}/up_{output_idx}_{mini_box_id}.png", up_image)

            if white_points_upper_threshold > np.count_nonzero(left_image) > white_points_lower_threshold:
                cv2.imwrite(f"{org_folder}/left_{output_idx}_{mini_box_id}.png", left_image)


def main():
    create_dataset_original_images()


if __name__ == "__main__":
    # TODO:
    # 1. Make sure "skel_np" folder is present in the root directory
    main()
