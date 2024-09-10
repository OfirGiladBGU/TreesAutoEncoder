import numpy as np
import cv2
import os
import pathlib
from tqdm import tqdm
import pandas as pd
from scipy.ndimage import label
from skimage import color

from dataset_list import DATASET_PATH, CROPPED_PATH
from dataset_utils import convert_nii_gz_to_numpy, convert_numpy_to_nii_gz


#########
# Utils #
#########
def connected_components_3d(data_3d: np.ndarray):
    # Define the structure for connectivity
    # Here, we use a structure that connects each voxel to its immediate neighbors
    structure = np.ones((3, 3, 3), dtype=np.int8)  # 26-connectivity

    # Label connected components
    labeled_array, num_features = label(data_3d, structure=structure)

    # print("Labeled Array:", labeled_array)
    # print("Number of Features:", num_features)

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
    if front is True:
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
    if back is True:
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
    if top is True:
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
    if bottom is True:
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
    if right is True:
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
    if left is True:
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


def crop_mini_cubes(data_3d: np.ndarray, size: tuple = (28, 28, 28), step: int = 14, cubes_data: bool = False):
    mini_cubes = list()
    mini_cubes_data = list()
    for x in range(0, data_3d.shape[0], step):
        for y in range(0, data_3d.shape[1], step):
            for z in range(0, data_3d.shape[2], step):
                # print(i, j, k)
                if (
                    x + size[0] > data_3d.shape[0] or
                    y + size[1] > data_3d.shape[1] or
                    z + size[2] > data_3d.shape[2]
                ):
                    continue

                start_x, start_y, start_z = x, y, z
                end_x, end_y, end_z = x + size[0], y + size[1], z + size[2]

                mini_cube = data_3d[start_x:end_x, start_y:end_y, start_z:end_z]
                mini_cubes.append(mini_cube)

                if cubes_data is True:
                    mini_cubes_data.append({
                        "start_x": start_x,
                        "start_y": start_y,
                        "start_z": start_z,
                        "end_x": end_x,
                        "end_y": end_y,
                        "end_z": end_z
                    })

    if cubes_data is False:
        return mini_cubes
    else:
        return mini_cubes, mini_cubes_data


def outlier_removal(pred_data: np.ndarray, label_data: np.ndarray):
    outlier_data = np.maximum(pred_data - label_data, 0)
    pred_data_fixed = pred_data - outlier_data
    return pred_data_fixed


# def image_missing_connected_components_removal(pred, label):
#     pred_missing_components = np.maximum(label - pred, 0)
#
#     threshold_image = pred_missing_components.astype(np.uint8)
#     threshold_image[threshold_image > 0] = 255
#     connectivity = 4
#     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(threshold_image, connectivity, cv2.CV_32S)
#
#     area_threshold = 10
#     for i in range(num_labels):
#         if stats[i, cv2.CC_STAT_AREA] < area_threshold:
#             # Set as 0 pixels of connected component with area smaller than 10 pixels
#             pred_missing_components[labels == i] = 0
#
#     # Remove from the label the missing connected components in the pred that are have area bigger than 10 pixels
#     repaired_label = label - pred_missing_components
#     return repaired_label


##################
# 2D Projections #
##################
def create_dataset_depth_2d_projections():
    # Inputs
    input_folders = {
        "labels": os.path.join(DATASET_PATH, "labels"),
        "preds": os.path.join(DATASET_PATH, "preds")
    }

    # Outputs
    output_folders = {
        "labels_2d": os.path.join(DATASET_PATH, "labels_2d"),
        "preds_2d": os.path.join(DATASET_PATH, "preds_2d")
    }

    # Create Output Folders
    for output_folder in output_folders.values():
        os.makedirs(output_folder, exist_ok=True)

    # Get the filepaths
    input_filepaths = dict()
    for key, value in input_folders.items():
        input_filepaths[key] = sorted(pathlib.Path(value).rglob("*.nii.gz"))

    filepaths_count = len(input_filepaths["labels"])
    for filepath_idx in tqdm(range(filepaths_count)):
        # Get index data:
        label_filepath = input_filepaths["labels"][filepath_idx]
        pred_filepath = input_filepaths["preds"][filepath_idx]

        output_idx = label_filepath.name.split(".nii.gz")[0]
        print(f"File: {output_idx}")

        label_numpy_data = convert_nii_gz_to_numpy(data_filepath=label_filepath)
        pred_numpy_data = convert_nii_gz_to_numpy(data_filepath=pred_filepath)

        label_projections = project_3d_to_2d(
            data_3d=label_numpy_data,
            front=True, back=True, top=True, bottom=True, left=True, right=True
        )
        pred_projections = project_3d_to_2d(
            data_3d=pred_numpy_data,
            front=True, back=True, top=True, bottom=True, left=True, right=True
        )

        images_6_views = ['top', 'bottom', 'front', 'back', 'left', 'right']
        for image_view in images_6_views:
            output_2d_format = f"{output_idx}_{image_view}"

            label_image = label_projections[f"{image_view}_image"]
            cv2.imwrite(os.path.join(output_folders["labels_2d"], f"{output_2d_format}.png"), label_image)

            pred_image = pred_projections[f"{image_view}_image"]
            cv2.imwrite(os.path.join(output_folders["preds_2d"], f"{output_2d_format}.png"), pred_image)


#################
# 3D Components #
#################
def build_preds_components():
    input_folder = os.path.join(DATASET_PATH, "preds")
    output_folder = os.path.join(DATASET_PATH, "preds_components")

    os.makedirs(output_folder, exist_ok=True)
    data_filepaths = sorted(pathlib.Path(input_folder).rglob("*.nii.gz"))

    filepaths_count = len(data_filepaths)
    for filepath_idx in tqdm(range(filepaths_count)):
        # Get index data:
        data_filepath = data_filepaths[filepath_idx]

        output_idx = data_filepath.name.split(".nii.gz")[0]

        numpy_data = convert_nii_gz_to_numpy(data_filepath=data_filepath)
        data_3d_components = connected_components_3d(data_3d=numpy_data)[0]

        # Save results
        save_name = os.path.join(output_folder, output_idx)
        convert_numpy_to_nii_gz(numpy_data=data_3d_components, save_name=save_name)


####################
# Original Dataset #
####################
def create_dataset_original_images():
    # Inputs
    input_folders = {
        "labels": os.path.join(DATASET_PATH, "labels"),
        "preds": os.path.join(DATASET_PATH, "preds"),
        "preds_components": os.path.join(DATASET_PATH, "preds_components")
    }

    # Outputs
    output_folders = {
        # Labels
        "labels_2d": os.path.join(CROPPED_PATH, "labels_2d_v6"),
        "labels_3d": os.path.join(CROPPED_PATH, "labels_3d_v6"),
        # Preds
        "preds_2d": os.path.join(CROPPED_PATH, "preds_2d_v6"),
        "preds_3d": os.path.join(CROPPED_PATH, "preds_3d_v6"),
        # Preds Components
        "preds_components_2d": os.path.join(CROPPED_PATH, "preds_components_2d_v6"),
        "preds_components_3d": os.path.join(CROPPED_PATH, "preds_components_3d_v6"),

        # Preds Fixed
        "preds_fixed_2d": os.path.join(CROPPED_PATH, "preds_fixed_2d_v6"),
        "preds_fixed_3d": os.path.join(CROPPED_PATH, "preds_fixed_3d_v6"),
        # Preds Fixed Components
        "preds_fixed_components_2d": os.path.join(CROPPED_PATH, "preds_fixed_components_2d_v6"),
        "preds_fixed_components_3d": os.path.join(CROPPED_PATH, "preds_fixed_components_3d_v6")
    }

    # Log
    log_filepath = os.path.join(CROPPED_PATH, "log.csv")
    log_data = dict()

    # Config
    size = (32, 32, 32)
    step = 16
    white_points_upper_threshold = size[0] * size[0] * 0.9
    white_points_lower_threshold = size[0] * size[0] * 0.1

    # Create Output Folders
    for output_folder in output_folders.values():
        os.makedirs(output_folder, exist_ok=True)

    # Get the filepaths
    input_filepaths = dict()
    filepaths_found = list()
    for key, value in input_folders.items():
        input_filepaths[key] = sorted(pathlib.Path(value).rglob("*.nii.gz"))
        filepaths_found.append(len(input_filepaths[key]))

    # Validation
    if len(set(filepaths_found)) != 1:
        raise ValueError("Different number of files found in the Input folders")

    filepaths_count = len(input_filepaths["labels"])
    for filepath_idx in range(filepaths_count):
        # Get index data:
        label_filepath = input_filepaths["labels"][filepath_idx]
        pred_filepath = input_filepaths["preds"][filepath_idx]
        pred_component_filepath = input_filepaths["preds_components"][filepath_idx]

        # Original data
        label_numpy_data = convert_nii_gz_to_numpy(data_filepath=label_filepath)
        pred_numpy_data = convert_nii_gz_to_numpy(data_filepath=pred_filepath)
        pred_component_numpy_data = convert_nii_gz_to_numpy(data_filepath=pred_component_filepath)

        # pred_fixed_numpy_data = np.logical_and(label_numpy_data, pred_numpy_data)
        pred_fixed_numpy_data = outlier_removal(pred_data=pred_numpy_data, label_data=label_numpy_data)
        pred_fixed_component_numpy_data = np.where(pred_fixed_numpy_data > 0.0, pred_component_numpy_data, 0.0)

        # Crop Mini Cubes
        label_cubes, cubes_data = crop_mini_cubes(data_3d=label_numpy_data, size=size, step=step, cubes_data=True)
        pred_cubes = crop_mini_cubes(data_3d=pred_numpy_data, size=size, step=step)
        pred_components_cubes = crop_mini_cubes(data_3d=pred_component_numpy_data, size=size, step=step)

        pred_fixed_cubes = crop_mini_cubes(data_3d=pred_fixed_numpy_data, size=size, step=step)
        pred_fixed_components_cubes = crop_mini_cubes(data_3d=pred_fixed_component_numpy_data, size=size, step=step)

        output_idx = label_filepath.name.split(".nii.gz")[0]
        print(
            f"File: {output_idx}\n"
            f"Total Mini Cubes: {len(label_cubes)}\n"
        )

        cubes_count = len(label_cubes)
        cubes_count_digits_count = len(str(cubes_count))
        for cube_idx in tqdm(range(cubes_count)):
            # Get index data:
            label_cube = label_cubes[cube_idx]
            pred_cube = pred_cubes[cube_idx]
            pred_components_cube = pred_components_cubes[cube_idx]

            pred_fixed_cube = pred_fixed_cubes[cube_idx]
            pred_fixed_components_cube = pred_fixed_components_cubes[cube_idx]

            # check that there are 2 or more components to connect
            global_components_3d_indices = list(np.unique(pred_components_cube))
            global_components_3d_indices.remove(0)
            global_components_3d_count = len(global_components_3d_indices)
            if global_components_3d_count < 2:
                continue

            # Project 3D to 2D (Preds)
            pred_projections = project_3d_to_2d(
                data_3d=pred_cube,
                component_3d=pred_components_cube,
                front=True, back=True, top=True, bottom=True, left=True, right=True
            )

            # Project 3D to 2D (Labels)
            label_projections = project_3d_to_2d(
                data_3d=label_cube,
                front=True, back=True, top=True, bottom=True, left=True, right=True
            )

            # Project 3D to 2D (Preds Fixed)
            pred_fixed_projections = project_3d_to_2d(
                data_3d=pred_fixed_cube,
                component_3d=pred_fixed_components_cube,
                front=True, back=True, top=True, bottom=True, left=True, right=True
            )

            images_6_views = ['top', 'bottom', 'front', 'back', 'left', 'right']
            condition1 = True
            condition2 = True
            for image_view in images_6_views:
                pred_image = pred_projections[f"{image_view}_image"]
                label_image = label_projections[f"{image_view}_image"]

                pred_fixed_image = pred_fixed_projections[f"{image_view}_image"]

                pred_components = pred_projections[f"{image_view}_components"]
                pred_projections[f"{image_view}_components"] = color.label2rgb(
                    label=np.where(pred_image > 0, pred_components, 0)
                ) * 255

                pred_fixed_components = pred_fixed_projections[f"{image_view}_components"]
                pred_fixed_projections[f"{image_view}_components"] = color.label2rgb(
                    label=np.where(pred_fixed_image > 0, pred_fixed_components, 0)
                ) * 255

                # Repair the labels - TODO: Check how to do smartly
                # label_projections[f"{image_6_view}_repaired_image"] = image_missing_connected_components_removal(
                #     pred=pred_image,
                #     label=label_image
                # )
                # repaired_label_image = label_projections[f"{image_6_view}_repaired_image"]

                # Check Condition 1 (If condition fails, skip the current mini cube):
                if not (white_points_upper_threshold > np.count_nonzero(pred_fixed_image) > white_points_lower_threshold):
                    condition1 = False
                    break

                # Check Condition 2 (If condition fails, skip the current mini cube):
                if not (white_points_upper_threshold > np.count_nonzero(label_image) > white_points_lower_threshold):
                    condition2 = False
                    break

            # DEBUG
            # if cube_idx == 3253:
            #     print("Debug")

            # Validate the conditions
            if not (condition1 is True and condition2 is True):
                continue

            cube_data = cubes_data[cube_idx]
            cube_idx_str = str(cube_idx).zfill(cubes_count_digits_count)

            for image_view in images_6_views:
                output_2d_format = f"{output_idx}_{cube_idx_str}_{image_view}"

                # 2D Folders - labels
                label_2d = label_projections[f"{image_view}_image"]
                cv2.imwrite(os.path.join(output_folders["labels_2d"], f"{output_2d_format}.png"), label_2d)

                # 2D Folder - preds
                pred_2d = pred_projections[f"{image_view}_image"]
                cv2.imwrite(os.path.join(output_folders["preds_2d"], f"{output_2d_format}.png"), pred_2d)

                # 2D Folders - preds components
                pred_components_2d = pred_projections[f"{image_view}_components"]
                cv2.imwrite(os.path.join(output_folders["preds_components_2d"], f"{output_2d_format}.png"), pred_components_2d)


                # 2D Folder - preds fixed
                pred_fixed_2d = pred_fixed_projections[f"{image_view}_image"]
                cv2.imwrite(os.path.join(output_folders["preds_fixed_2d"], f"{output_2d_format}.png"), pred_fixed_2d)

                # 2D Folders - preds fixed components
                pred_fixed_components_2d = pred_fixed_projections[f"{image_view}_components"]
                cv2.imwrite(os.path.join(output_folders["preds_fixed_components_2d"], f"{output_2d_format}.png"), pred_fixed_components_2d)

            output_3d_format = f"{output_idx}_{cube_idx_str}"

            # 3D Folders - labels
            save_name = os.path.join(output_folders["labels_3d"], output_3d_format)
            convert_numpy_to_nii_gz(numpy_data=label_cube, save_name=save_name)

            # 3D Folders - preds
            save_name = os.path.join(output_folders["preds_3d"], output_3d_format)
            convert_numpy_to_nii_gz(numpy_data=pred_cube, save_name=save_name)

            # 3D Folders - preds components
            save_name = os.path.join(output_folders["preds_components_3d"], output_3d_format)
            convert_numpy_to_nii_gz(numpy_data=pred_components_cube, save_name=save_name)


            # 3D Folders - preds fixed
            save_name = os.path.join(output_folders["preds_fixed_3d"], output_3d_format)
            convert_numpy_to_nii_gz(numpy_data=pred_fixed_cube, save_name=save_name)

            # 3D Folders - preds fixed components
            save_name = os.path.join(output_folders["preds_fixed_components_3d"], output_3d_format)
            convert_numpy_to_nii_gz(numpy_data=pred_fixed_components_cube, save_name=save_name)


            # Log 3D info
            label_components_cube = connected_components_3d(data_3d=label_cube)[0]

            local_components_3d_indices = list(np.unique(label_components_cube))
            local_components_3d_indices.remove(0)
            local_components_3d_count = len(local_components_3d_indices)

            cube_data.update({
                # "name": output_3d_format,
                "pred_global_components": global_components_3d_count,
                "label_local_components": local_components_3d_count
            })
            log_data[output_3d_format] = cube_data

            # DEBUG
            # convert_numpy_to_nii_gz(label_cube, save_name="1")
            # convert_numpy_to_nii_gz(pred_cube, save_name="2")

        if filepath_idx == 9:
            break

    pd.DataFrame(log_data).T.to_csv(log_filepath)


def main():
    # TODO: DEBUG
    # create_dataset_depth_2d_projections()

    build_preds_components()
    create_dataset_original_images()


if __name__ == "__main__":
    # TODO:
    # 1. Make sure "parse2022" folder is present in the root directory
    main()
