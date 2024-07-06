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


def convert_numpy_to_nii_gz(numpy_array, save_name="", save=False):
    ct_nii_gz = nib.Nifti1Image(numpy_array, affine=np.eye(4))
    if save and save_name != "":
        nib.save(ct_nii_gz, f"{save_name}.nii.gz")
    return ct_nii_gz


def _calculate_depth_projection(data_3d, axis):
    depth_projection = np.argmax(data_3d, axis=axis)
    max_projection = np.max(data_3d, axis=axis)
    axis_size = data_3d.shape[axis]

    return np.where(max_projection > 0,
                    (255 * (1 - (depth_projection / axis_size))).astype(int),
                    0)

    # return (255 * (1 - (depth_projection / axis_size))).astype(int)


def project_3d_to_2d(data_3d,
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

    # Front projection (XY plane)
    if front:
        flipped_data_3d = rotated_data_3d

        # Option 1
        # projections["front_image"] = np.max(data_3d, axis=2)

        # Option 2
        projections["front_image"] = _calculate_depth_projection(flipped_data_3d, axis=2)

    # Back projection (XY plane)
    if back:
        flipped_data_3d = rotated_data_3d
        flipped_data_3d = np.rot90(flipped_data_3d, k=2, axes=(1, 2))

        # Option 1
        # projections["back_image"] = np.max(flipped_data_3d, axis=2)

        # Option 2
        projections["back_image"] = _calculate_depth_projection(flipped_data_3d, axis=2)

    # Top projection (XZ plane)
    if top:
        flipped_data_3d = rotated_data_3d
        flipped_data_3d = np.rot90(flipped_data_3d, k=1, axes=(1, 2))

        # Option 1
        # projections["top_image"] = np.max(data_3d, axis=1)

        # Option 2
        projections["top_image"] = _calculate_depth_projection(flipped_data_3d, axis=0)

    # Bottom projection (XZ plane)
    if bottom:
        flipped_data_3d = rotated_data_3d
        flipped_data_3d = np.rot90(flipped_data_3d, k=1, axes=(1, 2))
        flipped_data_3d = np.rot90(flipped_data_3d, k=2, axes=(0, 1))

        # Option 1
        # projections["bottom_image"] = np.max(flipped_data_3d, axis=1)

        # Option 2
        projections["bottom_image"] = _calculate_depth_projection(flipped_data_3d, axis=0)

    # Right projection (YZ plane)
    if right:
        flipped_data_3d = rotated_data_3d
        flipped_data_3d = np.flip(flipped_data_3d, axis=1)

        # Option 1
        # projections["right_image"] = np.max(flipped_data_3d, axis=0)

        # Option 2
        projections["right_image"] = _calculate_depth_projection(flipped_data_3d, axis=1)

    # Left projection (YZ plane)
    if left:
        flipped_data_3d = rotated_data_3d
        flipped_data_3d = np.flip(flipped_data_3d, axis=1)
        flipped_data_3d = np.rot90(flipped_data_3d, k=2, axes=(1, 2))

        # Option 1
        # projections["left_image"] = np.max(data_3d, axis=0)

        # Option 2
        projections["left_image"] = _calculate_depth_projection(flipped_data_3d, axis=1)

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


##################
# 2D Projections #
##################
def create_dataset_depth_2d_projections():
    folder_path = "../parse2022/labels"
    org_folder = "./parse_labels_2d"

    # folder_path = "../parse2022/preds"
    # org_folder = "./parse_preds_2d"

    os.makedirs(org_folder, exist_ok=True)
    data_filepaths = os.listdir(folder_path)
    for data_filepath in data_filepaths:
        output_idx = data_filepath.split(".")[0]
        data_filepath = os.path.join(folder_path, data_filepath)
        ct_numpy = convert_nii_to_numpy(data_file=data_filepath)

        projection = project_3d_to_2d(ct_numpy,
                                      front=True, back=True, top=True, bottom=True, left=True, right=True)
        front_image = projection["front_image"]
        back_image = projection["back_image"]
        top_image = projection["top_image"]
        bottom_image = projection["bottom_image"]
        left_image = projection["left_image"]
        right_image = projection["right_image"]

        cv2.imwrite(f"{org_folder}/{output_idx}_front.png", front_image)
        cv2.imwrite(f"{org_folder}/{output_idx}_back.png", back_image)
        cv2.imwrite(f"{org_folder}/{output_idx}_top.png", top_image)
        cv2.imwrite(f"{org_folder}/{output_idx}_bottom.png", bottom_image)
        cv2.imwrite(f"{org_folder}/{output_idx}_left.png", left_image)
        cv2.imwrite(f"{org_folder}/{output_idx}_right.png", right_image)


####################
# Original Dataset #
####################
def create_dataset_original_images():
    folder_path1 = "../parse2022/preds"
    org_folder1 = "./parse_preds_mini_cropped_v3"

    folder_path2 = "../parse2022/labels"
    org_folder2 = "./parse_labels_mini_cropped_v3"

    size = (28, 28, 28)
    step = 14
    white_points_upper_threshold = size[0] * size[0] * 0.8
    white_points_lower_threshold = size[0] * size[0] * 0.1

    os.makedirs(org_folder1, exist_ok=True)
    os.makedirs(org_folder2, exist_ok=True)

    data_filepaths1 = sorted(os.listdir(folder_path1))
    data_filepaths2 = sorted(os.listdir(folder_path2))

    for batch_idx, (data_filepath1, data_filepath2) in enumerate(zip(data_filepaths1, data_filepaths2)):
        batch_idx += 1
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

        total_cubes_digits_count = len(str(len(mini_cubes1)))

        for mini_box_id, (mini_cube1, mini_cube2) in enumerate(zip(mini_cubes1, mini_cubes2)):
            projections1 = project_3d_to_2d(mini_cube1,
                                            front=True, back=True, top=True, bottom=True, left=True, right=True)
            front_image1 = projections1["front_image"]
            back_image1 = projections1["back_image"]
            top_image1 = projections1["top_image"]
            bottom_image1 = projections1["bottom_image"]
            left_image1 = projections1["left_image"]
            right_image1 = projections1["right_image"]

            projections2 = project_3d_to_2d(mini_cube2,
                                            front=True, back=True, top=True, bottom=True, left=True, right=True)
            front_image2 = projections2["front_image"]
            back_image2 = projections2["back_image"]
            top_image2 = projections2["top_image"]
            bottom_image2 = projections2["bottom_image"]
            left_image2 = projections2["left_image"]
            right_image2 = projections2["right_image"]

            # Repair the images
            front_image1 = image_outlier_removal(front_image1, front_image2)
            back_image2 = image_outlier_removal(back_image1, back_image2)
            top_image1 = image_outlier_removal(top_image1, top_image2)
            bottom_image1 = image_outlier_removal(bottom_image1, bottom_image2)
            left_image1 = image_outlier_removal(left_image1, left_image2)
            right_image1 = image_outlier_removal(right_image1, right_image2)

            condition1 = (
                white_points_upper_threshold > np.count_nonzero(front_image1) > white_points_lower_threshold and
                white_points_upper_threshold > np.count_nonzero(back_image1) > white_points_lower_threshold and
                white_points_upper_threshold > np.count_nonzero(top_image1) > white_points_lower_threshold and
                white_points_upper_threshold > np.count_nonzero(bottom_image1) > white_points_lower_threshold and
                white_points_upper_threshold > np.count_nonzero(left_image1) > white_points_lower_threshold and
                white_points_upper_threshold > np.count_nonzero(right_image1) > white_points_lower_threshold
            )
            condition2 = (
                white_points_upper_threshold > np.count_nonzero(front_image2) > white_points_lower_threshold and
                white_points_upper_threshold > np.count_nonzero(back_image2) > white_points_lower_threshold and
                white_points_upper_threshold > np.count_nonzero(top_image2) > white_points_lower_threshold and
                white_points_upper_threshold > np.count_nonzero(bottom_image2) > white_points_lower_threshold and
                white_points_upper_threshold > np.count_nonzero(left_image2) > white_points_lower_threshold and
                white_points_upper_threshold > np.count_nonzero(right_image2) > white_points_lower_threshold
            )

            if mini_box_id==3253 or condition1 and condition2:
                mini_box_id_str = str(mini_box_id).zfill(total_cubes_digits_count)
                # Folder1
                cv2.imwrite(f"{org_folder1}/{output_idx}_{mini_box_id_str}_front.png", front_image1)
                cv2.imwrite(f"{org_folder1}/{output_idx}_{mini_box_id_str}_back.png", back_image1)
                cv2.imwrite(f"{org_folder1}/{output_idx}_{mini_box_id_str}_top.png", top_image1)
                cv2.imwrite(f"{org_folder1}/{output_idx}_{mini_box_id_str}_bottom.png", bottom_image1)
                cv2.imwrite(f"{org_folder1}/{output_idx}_{mini_box_id_str}_left.png", left_image1)
                cv2.imwrite(f"{org_folder1}/{output_idx}_{mini_box_id_str}_right.png", right_image1)

                # Folder2
                cv2.imwrite(f"{org_folder2}/{output_idx}_{mini_box_id_str}_front.png", front_image2)
                cv2.imwrite(f"{org_folder2}/{output_idx}_{mini_box_id_str}_back.png", back_image2)
                cv2.imwrite(f"{org_folder2}/{output_idx}_{mini_box_id_str}_top.png", top_image2)
                cv2.imwrite(f"{org_folder2}/{output_idx}_{mini_box_id_str}_bottom.png", bottom_image2)
                cv2.imwrite(f"{org_folder2}/{output_idx}_{mini_box_id_str}_left.png", left_image2)
                cv2.imwrite(f"{org_folder2}/{output_idx}_{mini_box_id_str}_right.png", right_image2)

                convert_numpy_to_nii_gz(mini_cube1, save_name="1", save=True)
                convert_numpy_to_nii_gz(mini_cube2, save_name="2", save=True)

        if batch_idx == 1:
            break


def main():
    # create_dataset_depth_2d_projections()
    create_dataset_original_images()


if __name__ == "__main__":
    # TODO:
    # 1. Make sure "parse2022" folder is present in the root directory
    main()
