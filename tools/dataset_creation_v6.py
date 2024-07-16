import numpy as np
from scipy.ndimage import label
import nibabel as nib
import cv2
import os
from skimage import color

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


def connected_components_3d(data_3d):
    # Define the structure for connectivity
    # Here, we use a structure that connects each voxel to its immediate neighbors
    structure = np.ones((3, 3, 3), dtype=np.int8)  # 26-connectivity

    # Label connected components
    labeled_array, num_features = label(data_3d, structure=structure)

    # print("Labeled Array:")
    # print(labeled_array)
    print("Number of features:", num_features)

    return labeled_array, num_features


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


def project_3d_to_2d(data_3d,
                     component_3d=None,
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
    if front:
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
    if back:
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
    if top:
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
    if bottom:
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
    if right:
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
    if left:
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
    repaired_pred = pred - image_outlier
    return repaired_pred


def image_missing_connected_components_removal(pred, label):
    pred_missing_components = np.maximum(label - pred, 0)

    threshold_image = pred_missing_components.astype(np.uint8)
    threshold_image[threshold_image > 0] = 255
    connectivity = 4
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(threshold_image, connectivity, cv2.CV_32S)

    area_threshold = 10
    for i in range(num_labels):
        if stats[i, cv2.CC_STAT_AREA] < area_threshold:
            # Set as 0 pixels of connected component with area smaller than 10 pixels
            pred_missing_components[labels == i] = 0

    # Remove from the label the missing connected components in the pred that are have area bigger than 10 pixels
    repaired_label = label - pred_missing_components
    return repaired_label


#################
# 3D Components #
#################
def convert_to_3d_components():
    folder_path = "../parse2022/preds"
    org_folder = "../parse2022/preds_components"

    os.makedirs(org_folder, exist_ok=True)
    data_filepaths = os.listdir(folder_path)
    for data_filepath in data_filepaths:
        data_filepath = os.path.join(folder_path, data_filepath)
        ct_numpy = convert_nii_to_numpy(data_file=data_filepath)
        data_3d_components, _ = connected_components_3d(data_3d=ct_numpy)

        # Save results
        new_filepath = data_filepath.replace('preds', 'preds_components')
        output_save_name = os.path.splitext(os.path.splitext(new_filepath)[0])[0]  # To remove '.nii.gz'
        convert_numpy_to_nii_gz(numpy_array=data_3d_components, save_name=output_save_name, save=True)


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

        projection = project_3d_to_2d(data_3d=ct_numpy,
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
    org_folder1 = "./parse_preds_mini_cropped_v5"

    folder_path2 = "../parse2022/labels"
    org_folder2 = "./parse_labels_mini_cropped_v5"

    folder_path3 = "../parse2022/preds_components"
    org_folder3 = "./parse_preds_components_mini_cropped_v5"

    size = (32, 32, 32)
    step = 16
    white_points_upper_threshold = size[0] * size[0] * 0.9
    white_points_lower_threshold = size[0] * size[0] * 0.1

    os.makedirs(org_folder1, exist_ok=True)
    os.makedirs(org_folder2, exist_ok=True)
    os.makedirs(org_folder3, exist_ok=True)

    data_filepaths1 = sorted(os.listdir(folder_path1))
    data_filepaths2 = sorted(os.listdir(folder_path2))
    data_filepaths3 = sorted(os.listdir(folder_path3))

    for batch_idx in range(len(data_filepaths1)):
        data_filepath1 = data_filepaths1[batch_idx]
        data_filepath2 = data_filepaths2[batch_idx]
        data_filepath3 = data_filepaths3[batch_idx]

        batch_idx += 1
        output_idx = data_filepath1.split(".")[0]

        data_filepath1 = os.path.join(folder_path1, data_filepath1)
        data_filepath2 = os.path.join(folder_path2, data_filepath2)
        data_filepath3 = os.path.join(folder_path3, data_filepath3)

        ct_numpy1 = convert_nii_to_numpy(data_file=data_filepath1)
        ct_numpy2 = convert_nii_to_numpy(data_file=data_filepath2)
        ct_numpy3 = convert_nii_to_numpy(data_file=data_filepath3)

        cropped_data_3d_1 = ct_numpy1
        cropped_data_3d_2 = ct_numpy2
        cropped_data_3d_3 = ct_numpy3

        cropped_data_3d_1[cropped_data_3d_1 > 0] = 255
        cropped_data_3d_2[cropped_data_3d_2 > 0] = 255

        mini_cubes1 = crop_mini_cubes(cropped_data_3d=cropped_data_3d_1, size=size, step=step)
        mini_cubes2 = crop_mini_cubes(cropped_data_3d=cropped_data_3d_2, size=size, step=step)
        mini_cubes3 = crop_mini_cubes(cropped_data_3d=cropped_data_3d_3, size=size, step=step)

        print(
            f"File: {output_idx}\n"
            f"Total Mini Cubes 1: {len(mini_cubes1)}\n"
            f"Total Mini Cubes 2: {len(mini_cubes2)}\n"
            f"Total Mini Cubes 3: {len(mini_cubes3)}"
        )

        total_cubes_digits_count = len(str(len(mini_cubes1)))

        for mini_box_id in range(len(mini_cubes1)):
            mini_cube1 = mini_cubes1[mini_box_id]
            mini_cube2 = mini_cubes2[mini_box_id]
            mini_cube3 = mini_cubes3[mini_box_id]

            # check that there are 2 or more components to connect
            components_3d_indices = np.unique(mini_cube3)
            components_3d_indices = list(components_3d_indices)
            components_3d_indices.remove(0)
            components_3d_count = len(components_3d_indices)
            if components_3d_count < 2:
                continue

            # Project 3D to 2D (Preds)
            projections1 = project_3d_to_2d(data_3d=mini_cube1, component_3d=mini_cube3,
                                            front=True, back=True, top=True, bottom=True, left=True, right=True)
            front_image1 = projections1["front_image"]
            back_image1 = projections1["back_image"]
            top_image1 = projections1["top_image"]
            bottom_image1 = projections1["bottom_image"]
            left_image1 = projections1["left_image"]
            right_image1 = projections1["right_image"]

            front_components = projections1["front_components"]
            back_components = projections1["back_components"]
            top_components = projections1["top_components"]
            bottom_components = projections1["bottom_components"]
            left_components = projections1["left_components"]
            right_components = projections1["right_components"]

            # Project 3D to 2D (Labels)
            projections2 = project_3d_to_2d(data_3d=mini_cube2,
                                            front=True, back=True, top=True, bottom=True, left=True, right=True)
            front_image2 = projections2["front_image"]
            back_image2 = projections2["back_image"]
            top_image2 = projections2["top_image"]
            bottom_image2 = projections2["bottom_image"]
            left_image2 = projections2["left_image"]
            right_image2 = projections2["right_image"]

            # Repair the preds
            front_image1 = image_outlier_removal(front_image1, front_image2)
            back_image1 = image_outlier_removal(back_image1, back_image2)
            top_image1 = image_outlier_removal(top_image1, top_image2)
            bottom_image1 = image_outlier_removal(bottom_image1, bottom_image2)
            left_image1 = image_outlier_removal(left_image1, left_image2)
            right_image1 = image_outlier_removal(right_image1, right_image2)

            # Repair the components according to preds
            front_components = color.label2rgb(label=np.where(front_image1 > 0, front_components, 0)) * 255
            back_components = color.label2rgb(label=np.where(back_image1 > 0, back_components, 0)) * 255
            top_components = color.label2rgb(label=np.where(top_image1 > 0, top_components, 0)) * 255
            bottom_components = color.label2rgb(label=np.where(bottom_image1 > 0, bottom_components, 0)) * 255
            left_components = color.label2rgb(label=np.where(left_image1 > 0, left_components, 0)) * 255
            right_components = color.label2rgb(label=np.where(right_image1 > 0, right_components, 0)) * 255

            # Repair the labels
            front_image2 = image_missing_connected_components_removal(front_image1, front_image2)
            back_image2 = image_missing_connected_components_removal(back_image1, back_image2)
            top_image2 = image_missing_connected_components_removal(top_image1, top_image2)
            bottom_image2 = image_missing_connected_components_removal(bottom_image1, bottom_image2)
            left_image2 = image_missing_connected_components_removal(left_image1, left_image2)
            right_image2 = image_missing_connected_components_removal(right_image1, right_image2)

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

            # if mini_box_id==3253 or condition1 and condition2:

            if condition1 and condition2:
                mini_box_id_str = str(mini_box_id).zfill(total_cubes_digits_count)

                # Folder1 - preds
                cv2.imwrite(f"{org_folder1}/{output_idx}_{mini_box_id_str}_front.png", front_image1)
                cv2.imwrite(f"{org_folder1}/{output_idx}_{mini_box_id_str}_back.png", back_image1)
                cv2.imwrite(f"{org_folder1}/{output_idx}_{mini_box_id_str}_top.png", top_image1)
                cv2.imwrite(f"{org_folder1}/{output_idx}_{mini_box_id_str}_bottom.png", bottom_image1)
                cv2.imwrite(f"{org_folder1}/{output_idx}_{mini_box_id_str}_left.png", left_image1)
                cv2.imwrite(f"{org_folder1}/{output_idx}_{mini_box_id_str}_right.png", right_image1)

                # Folder2 - labels
                cv2.imwrite(f"{org_folder2}/{output_idx}_{mini_box_id_str}_front.png", front_image2)
                cv2.imwrite(f"{org_folder2}/{output_idx}_{mini_box_id_str}_back.png", back_image2)
                cv2.imwrite(f"{org_folder2}/{output_idx}_{mini_box_id_str}_top.png", top_image2)
                cv2.imwrite(f"{org_folder2}/{output_idx}_{mini_box_id_str}_bottom.png", bottom_image2)
                cv2.imwrite(f"{org_folder2}/{output_idx}_{mini_box_id_str}_left.png", left_image2)
                cv2.imwrite(f"{org_folder2}/{output_idx}_{mini_box_id_str}_right.png", right_image2)

                # Folder3 - components
                cv2.imwrite(f"{org_folder3}/{output_idx}_{mini_box_id_str}_front_components.png", front_components)
                cv2.imwrite(f"{org_folder3}/{output_idx}_{mini_box_id_str}_back_components.png", back_components)
                cv2.imwrite(f"{org_folder3}/{output_idx}_{mini_box_id_str}_top_components.png", top_components)
                cv2.imwrite(f"{org_folder3}/{output_idx}_{mini_box_id_str}_bottom_components.png", bottom_components)
                cv2.imwrite(f"{org_folder3}/{output_idx}_{mini_box_id_str}_left_components.png", left_components)
                cv2.imwrite(f"{org_folder3}/{output_idx}_{mini_box_id_str}_right_components.png", right_components)

                # convert_numpy_to_nii_gz(mini_cube1, save_name="1", save=True)
                # convert_numpy_to_nii_gz(mini_cube2, save_name="2", save=True)

        if batch_idx == 10:
            break


def main():
    # convert_to_3d_components()
    # create_dataset_depth_2d_projections()
    create_dataset_original_images()


if __name__ == "__main__":
    # TODO:
    # 1. Make sure "parse2022" folder is present in the root directory
    main()
