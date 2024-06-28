import numpy as np
import nibabel as nib
import cv2
import os


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
        # Option 1
        # projections["front_image"] = np.max(data_3d, axis=2)

        # Option 2
        depth_projection = np.argmax(data_3d, axis=2)
        max_projection = np.max(data_3d, axis=2)
        axis_size = data_3d.shape[2]

        projections["front_image"] = np.where(max_projection > 0,
                                              (255 * (1 - (depth_projection / axis_size))).astype(int),
                                              0)

    # Up projection (XZ plane)
    if up:
        if up_cut != -1:
            data_3d = data_3d[:, 0:up_cut, :]
        # Option 1
        # projections["up_image"] = np.max(data_3d, axis=1)

        # Option 2
        depth_projection = np.argmax(data_3d, axis=1)
        max_projection = np.max(data_3d, axis=1)
        axis_size = data_3d.shape[1]

        projections["up_image"] = np.where(max_projection > 0,
                                           (255 * (1 - (depth_projection / axis_size))).astype(int),
                                           0)

    # Left projection (YZ plane)
    if left:
        if left_cut != -1:
            data_3d = data_3d[0:left_cut, :, :]
        # Option 1
        # projections["left_image"] = np.max(data_3d, axis=0)

        # Option 2
        depth_projection = np.argmax(data_3d, axis=0)
        max_projection = np.max(data_3d, axis=0)
        axis_size = data_3d.shape[0]

        projections["left_image"] = np.where(max_projection > 0,
                                             (255 * (1 - (depth_projection / axis_size))).astype(int),
                                             0)

    return projections


def crop_mini_cubes(cropped_data_3d, size=(28, 28, 28), step=14):
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


def image_outlier_removal(pred, label):
    image_outlier = np.maximum(pred - label, 0)
    repaired_image = pred - image_outlier
    return repaired_image


####################
# Original Dataset #
####################
def create_dataset_original_images():
    folder_path1 = "../parse2022/preds"
    org_folder1 = "./parse_preds_mini_cropped_v2"

    folder_path2 = "../parse2022/labels"
    org_folder2 = "./parse_labels_mini_cropped_v2"

    size = (64, 64, 64)
    step = 32
    white_points_upper_threshold = 64 * 64 * 0.8
    white_points_lower_threshold = 64 * 64 * 0.1

    os.makedirs(org_folder1, exist_ok=True)
    os.makedirs(org_folder2, exist_ok=True)

    data_filepaths1 = sorted(os.listdir(folder_path1))
    data_filepaths2 = sorted(os.listdir(folder_path2))

    for data_filepath1, data_filepath2 in zip(data_filepaths1, data_filepaths2):
        output_idx = data_filepath1.split(".")[0]

        data_filepath1 = os.path.join(folder_path1, data_filepath1)
        data_filepath2 = os.path.join(folder_path2, data_filepath2)

        ct_numpy1 = convert_nii_to_numpy(data_file=data_filepath1)
        ct_numpy2 = convert_nii_to_numpy(data_file=data_filepath2)

        cropped_data_3d_1 = ct_numpy1
        cropped_data_3d_2 = ct_numpy2

        cropped_data_3d_1[cropped_data_3d_1 > 0] = 255
        cropped_data_3d_2[cropped_data_3d_2 > 0] = 255

        mini_cubes1 = crop_mini_cubes(cropped_data_3d_1, size=size, step=step)
        mini_cubes2 = crop_mini_cubes(cropped_data_3d_2, size=size, step=step)

        print(
            f"File: {output_idx}\n"
            f"Total Mini Cubes 1: {len(mini_cubes1)}\n"
            f"Total Mini Cubes 2: {len(mini_cubes2)}"
        )

        for mini_box_id, (mini_cube1, mini_cube2) in enumerate(zip(mini_cubes1, mini_cubes2)):
            projections1 = project_3d_to_2d(mini_cube1, front=True, up=True, left=True)
            front_image1 = projections1["front_image"]
            up_image1 = projections1["up_image"]
            left_image1 = projections1["left_image"]

            projections2 = project_3d_to_2d(mini_cube2, front=True, up=True, left=True)
            front_image2 = projections2["front_image"]
            up_image2 = projections2["up_image"]
            left_image2 = projections2["left_image"]

            # Repair the images
            front_image1 = image_outlier_removal(front_image1, front_image2)
            up_image1 = image_outlier_removal(up_image1, up_image2)
            left_image1 = image_outlier_removal(left_image1, left_image2)

            condition1 = (
                white_points_upper_threshold > np.count_nonzero(front_image1) > white_points_lower_threshold and
                white_points_upper_threshold > np.count_nonzero(up_image1) > white_points_lower_threshold and
                white_points_upper_threshold > np.count_nonzero(left_image1) > white_points_lower_threshold
            )
            condition2 = (
                white_points_upper_threshold > np.count_nonzero(front_image2) > white_points_lower_threshold and
                white_points_upper_threshold > np.count_nonzero(up_image2) > white_points_lower_threshold and
                white_points_upper_threshold > np.count_nonzero(left_image2) > white_points_lower_threshold
            )

            if condition1 and condition2:
                # Folder1
                cv2.imwrite(f"{org_folder1}/{output_idx}_{mini_box_id}_front.png", front_image1)
                cv2.imwrite(f"{org_folder1}/{output_idx}_{mini_box_id}_up.png", up_image1)
                cv2.imwrite(f"{org_folder1}/{output_idx}_{mini_box_id}_left.png", left_image1)

                # Folder2
                cv2.imwrite(f"{org_folder2}/{output_idx}_{mini_box_id}_front.png", front_image2)
                cv2.imwrite(f"{org_folder2}/{output_idx}_{mini_box_id}_up.png", up_image2)
                cv2.imwrite(f"{org_folder2}/{output_idx}_{mini_box_id}_left.png", left_image2)

        # break


def main():
    create_dataset_original_images()


if __name__ == "__main__":
    # TODO:
    # 1. Make sure "parse2022" folder is present in the root directory
    main()
