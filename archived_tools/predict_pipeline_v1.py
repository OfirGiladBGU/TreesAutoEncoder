import argparse
import os
import pathlib
import torch
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from scipy.ndimage import convolve, label
from typing import Tuple
from concurrent.futures import ThreadPoolExecutor
import datetime

from datasets_forge.dataset_configurations import *
from datasets.dataset_utils import *
from models.model_list import init_model
# TODO: Debug Tools
from datasets_visualize.dataset_visulalization import interactive_plot_2d, interactive_plot_3d

#########
# Utils #
#########
def preprocess_2d(data_3d_filepath: str,
                  data_2d_folder: str,
                  apply_batch_merge: bool = False) -> torch.Tensor:
    data_3d_basename = get_data_file_stem(data_filepath=data_3d_filepath)
    data_2d_basename = f"{data_3d_basename}_<VIEW>"

    # # Get relative path parts
    # relative_filepath = data_3d_filepath.relative_to(CROPS_PATH)
    # relative_filepath_parts = list(relative_filepath.parts)
    #
    # # Update relative path parts to the relevant 2D images path
    # relative_filepath_parts[0] = relative_filepath_parts[0].replace("3d", "2d")
    # relative_filepath_parts[-1] = f"{data_2d_basename}.png"
    # format_of_2d_images_relative_filepath = pathlib.Path(*relative_filepath_parts)
    # format_of_2d_images = os.path.join(CROPS_PATH, format_of_2d_images_relative_filepath)

    format_of_2d_images = os.path.join(data_2d_folder, data_2d_basename + ".png")

    # Projections 2D
    data_2d_list = list()
    for image_view in IMAGES_6_VIEWS:
        image_path = format_of_2d_images.replace("<VIEW>", image_view)
        image_data = convert_data_file_to_numpy(data_filepath=image_path)
        torch_image = transforms.ToTensor()(image_data)
        if apply_batch_merge is True:
            torch_image = torch_image.squeeze(0)
        data_2d_list.append(torch_image)

    # Shape: (1, 6, w, h)
    if apply_batch_merge is True:
        data_2d_input = torch.stack(data_2d_list).unsqueeze(0)
    # Shape: (6, 1, w, h)
    else:
        data_2d_input = torch.stack(data_2d_list)
    return data_2d_input


def postprocess_2d(data_3d_filepath: str,
                   data_2d_input: torch.Tensor,
                   data_2d_output: torch.Tensor,
                   apply_input_merge: bool = False,
                   log_data: pd.DataFrame = None) -> Tuple[torch.Tensor, torch.Tensor]:
    data_3d_basename = get_data_file_stem(data_filepath=data_3d_filepath)

    # Convert (1, 6, w, h) to (6, w, h)
    if data_2d_input.shape[0] == 1 and data_2d_input.shape[1] == 6:
        data_2d_input = data_2d_input.squeeze(0)
        data_2d_output = data_2d_output.squeeze(0)
    # Convert (6, 1, w, h) to (6, w, h)
    elif data_2d_input.shape[0] == 6 and data_2d_input.shape[1] == 1:
        data_2d_input = data_2d_input.squeeze(1)
        data_2d_output = data_2d_output.squeeze(1)
    # Apply no change to (6, w, h)
    elif data_2d_input.shape[0] == 6 and len(data_2d_input.shape) == 3:
        pass
    else:
        raise ValueError("Invalid shape")

    # TODO: Threshold
    apply_threshold(tensor=data_2d_output, threshold=0.2, keep_values=True)

    if log_data is not None:
        # Replace invalid data with original input
        for idx, image_view in enumerate(IMAGES_6_VIEWS):
            view_advance_valid = f"{image_view}_advance_valid"
            if view_advance_valid in log_data.columns:
                data_2d_matching_rows = log_data[log_data[log_data.columns[0]] == data_3d_basename]
                data_2d_row = list(data_2d_matching_rows.iterrows())[0][1]
                if data_2d_row[view_advance_valid] is False:
                    data_2d_output[idx] = data_2d_input[idx].clone()
                else:
                    pass
            else:
                # print(f"No '{view_advance_valid}' column in log csv data")
                pass  # For Eval data there is no advance_valid column
    else:  # TODO: use classifier results
        pass

    if apply_input_merge is True:
        data_2d_output = data_2d_input + torch.where(data_2d_input == 0, data_2d_output, 0)

    return data_2d_input, data_2d_output


def debug_2d(data_3d_filepath: str, data_2d_input: torch.Tensor, data_2d_output: torch.Tensor):
    data_3d_basename = get_data_file_stem(data_filepath=data_3d_filepath)

    data_2d_input_copy = data_2d_input.clone().numpy()
    data_2d_output_copy = data_2d_output.clone().numpy()

    columns = 6
    rows = 2
    fig = plt.figure(figsize=(columns + 0.5, rows + 0.5))
    ax = list()

    # 2D Input
    for j in range(columns):
        ax.append(fig.add_subplot(rows, columns, 0 * columns + j + 1))
        numpy_image = data_2d_input_copy[j]
        numpy_image = numpy_image * 255
        numpy_image = numpy_image.astype(np.uint8)
        numpy_image = np.expand_dims(numpy_image, axis=-1)
        plt.imshow(numpy_image, cmap='gray')
        ax[j].set_title(f"View {IMAGES_6_VIEWS[j]}:")

    # 2D Output
    for j in range(columns):
        ax.append(fig.add_subplot(rows, columns, 1 * columns + j + 1))
        numpy_image = data_2d_output_copy[j]
        numpy_image = numpy_image * 255
        numpy_image = numpy_image.astype(np.uint8)
        numpy_image = np.expand_dims(numpy_image, axis=-1)
        plt.imshow(numpy_image, cmap='gray')
        ax[j].set_title(f"View {IMAGES_6_VIEWS[j]}:")

    save_path = os.path.join(PREDICT_PIPELINE_RESULTS_PATH, "output_2d")
    os.makedirs(save_path, exist_ok=True)
    save_filepath = os.path.join(save_path, data_3d_basename)
    fig.tight_layout()
    plt.savefig(save_filepath)
    plt.close(fig)


def naive_noise_filter(data_3d_original: np.ndarray, data_3d_input: np.ndarray):
    # V1

    # Define a 3x3x3 kernel that will be used to check 6 neighbors (left, right, up, down, front, back)
    kernel = np.zeros((3, 3, 3), dtype=int)

    # Set only the 6-connectivity neighbors in the kernel
    kernel[1, 0, 1] = 1  # Left
    kernel[1, 2, 1] = 1  # Right
    kernel[0, 1, 1] = 1  # Up
    kernel[2, 1, 1] = 1  # Down
    kernel[1, 1, 0] = 1  # Front
    kernel[1, 1, 2] = 1  # Back

    # Convolve the binary array with the kernel to count neighbors
    neighbors_count = convolve(data_3d_input, kernel, mode='constant', cval=0)

    # Filter out voxels that have no neighboring voxels (i.e., neighbors_count == 0)
    filtered_data_3d_input = np.where((data_3d_input == 1) & (neighbors_count > 0), 1, 0)

    return filtered_data_3d_input

    # filtered_data_3d = data_3d_original.copy().astype(np.uint8)
    # data_delta = (data_3d_input - data_3d_original > 0.5).astype(np.uint8)
    #
    # # Identify connected components in data_delta
    # delta_labeled, delta_num_components = connected_components_3d(data_3d=data_delta, connectivity_type=6)
    #
    # # Iterate through connected components in data_delta
    # for component_label in range(1, delta_num_components + 1):
    #     # Create a mask for the current connected component
    #     component_mask = np.equal(delta_labeled, component_label).astype(np.uint8)
    #
    #     # Check the number of connected components before adding the mask
    #     components_before = connected_components_3d(filtered_data_3d)[1]
    #
    #     # Create a temporary data with the component added
    #     temp_fix = np.logical_or(filtered_data_3d, component_mask)
    #     components_after = connected_components_3d(temp_fix)[1]
    #
    #     # Add the component only if it does decrease the number of connected components
    #     if components_after < components_before:
    #         filtered_data_3d = temp_fix
    #
    # filtered_data_3d = filtered_data_3d.astype(np.float32)
    # return filtered_data_3d


def global_components_noise_filter(data_3d_original: np.ndarray, data_3d_input: np.ndarray,
                                   base_mode: int = 0, connectivity_type: int = 26):
    filtered_data_3d = data_3d_original.copy().astype(np.uint8)

    # Identify connected components in delta_cube
    if base_mode == 0:
        delta_data = (data_3d_input - data_3d_original > 0.5).astype(np.uint8)

        delta_labeled, delta_num_components = connected_components_3d(
            data_3d=delta_data,
            connectivity_type=connectivity_type
        )
    else:
        delta_labeled, delta_num_components = connected_components_3d(
            data_3d=data_3d_input,
            connectivity_type=connectivity_type
        )

    # Iterate through connected components in delta_cube
    for component_label in range(1, delta_num_components + 1):
        # Create a mask for the current connected component
        component_mask = np.equal(delta_labeled, component_label).astype(np.uint8)

        # Check the number of connected components before adding the mask
        components_before = connected_components_3d(filtered_data_3d)[1]

        # Create a temporary data with the component added
        temp_fix = np.logical_or(filtered_data_3d, component_mask)
        components_after = connected_components_3d(temp_fix)[1]

        # Add the component only if it does not increase the number of connected components
        if base_mode == 0:
            if not (components_before <= components_after):
                filtered_data_3d = temp_fix
        else:
            if not (components_before < components_after):
                filtered_data_3d = temp_fix

    filtered_data_3d = filtered_data_3d.astype(data_3d_original.dtype)
    return filtered_data_3d


def local_components_noise_filter(data_3d_original: np.ndarray, data_3d_input: np.ndarray,
                                  connectivity_type: int = 26):
    # TODO: Check completion in the local scope
    padding_size = 2

    filtered_data_3d = data_3d_original.copy().astype(np.uint8)

    # Calculate the connected components for the delta cube
    delta_cube = ((data_3d_input - data_3d_original) > 0.5).astype(np.uint8)

    # Identify connected components in delta_cube
    delta_labeled, delta_num_components = connected_components_3d(
        data_3d=delta_cube,
        connectivity_type=connectivity_type
    )

    # Iterate through connected components in delta_cube
    for component_label in range(1, delta_num_components + 1):
        # Create a mask for the current connected component
        component_mask = np.equal(delta_labeled, component_label).astype(np.uint8)

        # ROI - cropped area between the component mask: top, bottom, left, right, front, back
        coords = np.argwhere(component_mask > 0)

        # Get bounding box in 3D
        top = np.min(coords[:, 0])  # Minimum row index (Y-axis)
        bottom = np.max(coords[:, 0])  # Maximum row index (Y-axis)

        left = np.min(coords[:, 1])  # Minimum column index (X-axis)
        right = np.max(coords[:, 1])  # Maximum column index (X-axis)

        front = np.min(coords[:, 2])  # Minimum depth index (Z-axis)
        back = np.max(coords[:, 2])  # Maximum depth index (Z-axis)

        # Ensure ROI is within valid bounds (2 voxels padding)
        min_y = max(0, top - padding_size)
        max_y = min(bottom + padding_size + 1, filtered_data_3d.shape[0])

        min_x = max(0, left - padding_size)
        max_x = min(right + padding_size + 1, filtered_data_3d.shape[1])

        min_z = max(0, front - padding_size)
        max_z = min(back + padding_size + 1, filtered_data_3d.shape[2])

        # Check the number of connected components before adding the mask
        roi_temp_before = filtered_data_3d[min_y:max_y, min_x:max_x, min_z:max_z]
        components_before = connected_components_3d(data_3d=roi_temp_before, connectivity_type=6)[1]

        # Create a temporary image with the component added
        temp_fix = np.logical_or(filtered_data_3d, component_mask)
        roi_temp_after = temp_fix[min_y:max_y, min_x:max_x, min_z:max_z]
        components_after = connected_components_3d(data_3d=roi_temp_after, connectivity_type=6)[1]

        # Add the component only if it does not increase the number of connected components
        # (on the local scope)
        if not (components_before < components_after):
            filtered_data_3d = temp_fix

        # if not (components_before =< components_after):
        #     filtered_data_3d = temp_fix

    filtered_data_3d = filtered_data_3d.astype(data_3d_original.dtype)
    return filtered_data_3d


def preprocess_3d(data_3d_filepath: str,
                  data_2d_output: torch.Tensor,
                  apply_fusion: bool = False,
                  apply_noise_filter: bool = False) -> torch.Tensor:
    pred_3d = convert_data_file_to_numpy(data_filepath=data_3d_filepath)
    data_2d_output = data_2d_output.numpy()

    # Reconstruct 3D
    data_3d_list = list()
    for idx, image_view in enumerate(IMAGES_6_VIEWS):
        numpy_image = data_2d_output[idx] * 255
        data_3d = reverse_rotations(numpy_image=numpy_image, view_type=image_view, source_data_filepath=data_3d_filepath)
        data_3d_list.append(data_3d)

    data_3d_reconstruct = data_3d_list[0]
    for i in range(1, len(data_3d_list)):
        data_3d_reconstruct = np.logical_or(data_3d_reconstruct, data_3d_list[i])
    data_3d_reconstruct = data_3d_reconstruct.astype(np.float32)

    # Fusion 3D
    if apply_fusion is True:
        data_3d_fusion = np.logical_or(data_3d_reconstruct, pred_3d)
        data_3d_fusion = data_3d_fusion.astype(np.float32)
        data_3d_input = data_3d_fusion
    else:
        data_3d_input = data_3d_reconstruct

    # TODO: validate filters!!!
    # TODO: add the noise filters as revers continuity fix for both 2D and 3D!
    if apply_noise_filter is True:
        # Parse2022
        # data_3d_input = naive_noise_filter(data_3d_original=pred_3d, data_3d_input=data_3d_input)

        # data_3d_input = global_components_noise_filter(
        #     data_3d_original=pred_3d, data_3d_input=data_3d_input, base_mode=0, connectivity_type=6
        # )

        # Pipes3D
        data_3d_input = global_components_noise_filter(
            data_3d_original=pred_3d, data_3d_input=data_3d_input, base_mode=1, connectivity_type=6
        )

        # data_3d_input = local_components_noise_filter(
        #     data_3d_original=pred_3d, data_3d_input=data_3d_input, connectivity_type=26
        # )

    data_3d_input = torch.Tensor(data_3d_input).unsqueeze(0).unsqueeze(0)
    return data_3d_input


def postprocess_3d(data_3d_input: torch.Tensor,
                   data_3d_output: torch.Tensor,
                   apply_input_merge: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    data_3d_input = data_3d_input.squeeze().squeeze()
    data_3d_output = data_3d_output.squeeze().squeeze()

    # TODO: Threshold
    apply_threshold(tensor=data_3d_output, threshold=0.5, keep_values=False)

    if apply_input_merge is True:
        data_3d_output = data_3d_input + torch.where(data_3d_input == 0, data_3d_output, 0)

    return data_3d_input, data_3d_output


def debug_3d(data_3d_filepath, data_3d_input: torch.Tensor):
    data_3d_basename = get_data_file_stem(data_filepath=data_3d_filepath)

    data_3d_input = data_3d_input.clone().numpy()

    save_path = os.path.join(PREDICT_PIPELINE_RESULTS_PATH, "output_3d")
    os.makedirs(save_path, exist_ok=True)
    save_filepath = os.path.join(save_path, f"{data_3d_basename}_input")
    convert_numpy_to_data_file(numpy_data=data_3d_input, source_data_filepath=data_3d_filepath,
                               save_filename=save_filepath)


def export_output(data_3d_filepath, data_3d_output: torch.Tensor):
    data_3d_basename = get_data_file_stem(data_filepath=data_3d_filepath)

    data_3d_output = data_3d_output.numpy()

    save_path = os.path.join(PREDICT_PIPELINE_RESULTS_PATH, "output_3d")
    os.makedirs(save_path, exist_ok=True)

    save_filepath = os.path.join(save_path, f"{data_3d_basename}_output")
    convert_numpy_to_data_file(numpy_data=data_3d_output, source_data_filepath=data_3d_filepath,
                               save_filename=save_filepath)


##################
# Core Functions #
##################
def init_pipeline_models():
    # Load models
    filepath, ext = os.path.splitext(args.weights_filepath)

    # Load 2D model
    if len(args.model_2d) > 0:
        args.input_size = args.input_size_model_2d
        args.model = args.model_2d
        model_2d = init_model(args=args)
        model_2d_weights_filepath = f"{filepath}_{DATASET_OUTPUT_FOLDER}_{model_2d.model_name}{ext}"
        # TODO: model_2d_weights_filepath = f"{filepath}_PipeForge3DPCD_{model_2d.model_name}{ext}"
        model_2d.load_state_dict(torch.load(model_2d_weights_filepath))
        model_2d.eval()
        model_2d.to(args.device)
        args.model_2d_class = model_2d
    else:
        pass

    # Load 3D model
    if len(args.model_3d) > 0:
        args.input_size = args.input_size_model_3d
        args.model = args.model_3d
        model_3d = init_model(args=args)
        model_3d_weights_filepath = f"{filepath}_{DATASET_OUTPUT_FOLDER}_{model_3d.model_name}{ext}"
        # TODO: model_3d_weights_filepath = f"{filepath}_PipeForge3DPCD_{model_3d.model_name}{ext}"
        model_3d.load_state_dict(torch.load(model_3d_weights_filepath))
        model_3d.eval()
        model_3d.to(args.device)
        args.model_3d_class = model_3d
    else:
        pass


def single_predict(data_3d_filepath, data_2d_folder, log_data=None, enable_debug=True):
    # CONFIGS
    apply_input_merge_2d = True  # False - for PipeForge3DPCD
    apply_input_merge_3d = True
    apply_fusion = True
    apply_noise_filter = True

    # INPUTS
    data_3d_filepath = str(data_3d_filepath)
    data_2d_folder = str(data_2d_folder)

    os.makedirs(PREDICT_PIPELINE_RESULTS_PATH, exist_ok=True)

    if args.input_size_model_2d[0] == 6 and len(args.input_size_model_2d) == 3:
        apply_batch_merge = True
    else:
        apply_batch_merge = False

    with torch.no_grad():
        # TODO: Support 1D model TBD

        ##############
        # 2D Section #
        ##############
        data_2d_input = preprocess_2d(
            data_3d_filepath=data_3d_filepath,
            data_2d_folder=data_2d_folder,
            apply_batch_merge=apply_batch_merge
        )

        # Predict 2D
        if len(args.model_2d) > 0:
            data_2d_output = args.model_2d_class(data_2d_input)

            # Handle additional tasks
            if "confidence map" in getattr(args.model_2d_class, "additional_tasks", list()):
                data_2d_output, data_2d_output_confidence = data_2d_output
                data_2d_output = torch.where(data_2d_output_confidence > 0.5, data_2d_output, 0)
        else:
            data_2d_output = data_2d_input.clone()

        (data_2d_input, data_2d_output) = postprocess_2d(
            data_3d_filepath=data_3d_filepath,
            data_2d_input=data_2d_input,
            data_2d_output=data_2d_output,
            apply_input_merge=apply_input_merge_2d,
            log_data=log_data
        )

        # DEBUG
        if enable_debug is True:
            debug_2d(data_3d_filepath=data_3d_filepath, data_2d_input=data_2d_input, data_2d_output=data_2d_output)

        ##############
        # 3D Section #
        ##############
        data_3d_input = preprocess_3d(
            data_3d_filepath=data_3d_filepath,
            data_2d_output=data_2d_output,
            apply_fusion=apply_fusion,
            apply_noise_filter=apply_noise_filter
        )

        # Predict 3D
        if len(args.model_3d) > 0:
            data_3d_output = args.model_3d_class(data_3d_input)
        else:
            data_3d_output = data_3d_input.clone()

        (data_3d_input, data_3d_output) = postprocess_3d(
            data_3d_input=data_3d_input,
            data_3d_output=data_3d_output,
            apply_input_merge=apply_input_merge_3d
        )

        # DEBUG
        if enable_debug is True:
            debug_3d(data_3d_filepath=data_3d_filepath, data_3d_input=data_3d_input)

        export_output(data_3d_filepath=data_3d_filepath, data_3d_output=data_3d_output)


def test_single_predict():
    data_type = DataType.TRAIN
    # data_type = DataType.EVAL

    # input_filename = "PA000005_11899.nii.gz"
    # input_filename = "PA000078_11996.nii.gz"
    # input_filename = "47_52.npy"
    input_filename = "PA000005_0650.nii.gz"

    if data_type == DataType.TRAIN:
        log_data = pd.read_csv(TRAIN_LOG_PATH)

        # data_3d_folder = PREDS_3D
        # data_2d_folder = PREDS_2D

        data_3d_folder = PREDS_FIXED_3D
        data_2d_folder = PREDS_FIXED_2D

        # data_3d_folder = PREDS_ADVANCED_FIXED_3D
        # data_2d_folder = PREDS_ADVANCED_FIXED_2D
    else:
        log_data = pd.read_csv(EVAL_LOG_PATH)

        data_3d_folder = EVALS_3D
        data_2d_folder = EVALS_3D

    # Get filepaths
    data_3d_filepath = os.path.join(data_3d_folder, input_filename)
    single_predict(
        data_3d_filepath=data_3d_filepath,
        data_2d_folder=data_2d_folder,
        log_data=log_data
    )


def full_predict(data_3d_basename, data_type: DataType):
    start_time = datetime.datetime.now()
    start_timestamp = start_time.strftime('%Y-%m-%d_%H-%M-%S')
    print(
        f"[Full Predict] Started Predict... "
        f"(Timestamp: {start_timestamp})"
    )

    if data_type == DataType.TRAIN:
        log_data = pd.read_csv(TRAIN_LOG_PATH)

        data_3d_folder = PREDS_3D
        data_2d_folder = PREDS_2D

        # data_3d_folder = PREDS_FIXED_3D
        # data_2d_folder = PREDS_FIXED_2D

        # data_3d_folder = PREDS_ADVANCED_FIXED_3D
        # data_2d_folder = PREDS_ADVANCED_FIXED_2D
    else:
        log_data = pd.read_csv(EVAL_LOG_PATH)

        data_3d_folder = EVALS_3D
        data_2d_folder = EVALS_3D

    # Get filepaths
    data_3d_filepaths = pathlib.Path(data_3d_folder).rglob(f"{data_3d_basename}_*.*")
    data_3d_filepaths = sorted(data_3d_filepaths)

    # Single-threading - Sequential
    # for data_3d_filepath in tqdm(data_3d_filepaths):
    #     single_predict(
    #         data_3d_filepath=data_3d_filepath,
    #         data_2d_folder=data_2d_folder,
    #         log_data=log_data,
    #         enable_debug=True
    #     )

    # Multi-threading
    futures = []
    with ThreadPoolExecutor() as executor:
        # Submit all tasks
        for data_3d_filepath in data_3d_filepaths:
            futures.append(
                executor.submit(
                    single_predict,
                    data_3d_filepath=data_3d_filepath,
                    data_2d_folder=data_2d_folder,
                    log_data=log_data,
                    enable_debug=False
                )
            )
        # "Join" on all tasks by waiting for each future to complete.
        for future in tqdm(futures):
            future.result()  # This will block until the future is done.

    end_time = datetime.datetime.now()
    end_timestamp = end_time.strftime('%Y-%m-%d_%H-%M-%S')
    print(
        f"[Full Predict] Completed Predict... "
        f"(Timestamp: {end_timestamp}, Full Predict Time Elapsed: {end_time - start_time})"
    )


def full_merge(data_3d_basename, data_type: DataType):
    start_time = datetime.datetime.now()
    start_timestamp = start_time.strftime('%Y-%m-%d_%H-%M-%S')
    print(
        f"[Full Merge] Started Merge... "
        f"(Timestamp: {start_timestamp})"
    )

    if data_type == DataType.TRAIN:
        # TODO: create csv log per 3D object to improve search (Maybe in the future)
        log_data = pd.read_csv(TRAIN_LOG_PATH)

        # data_3d_folder = PREDS
        data_3d_folder = PREDS_FIXED

    else:
        log_data = pd.read_csv(EVAL_LOG_PATH)

        data_3d_folder = EVALS

    # Input 3D object
    data_3d_filepath = list(pathlib.Path(data_3d_folder).rglob(f"{data_3d_basename}*"))
    if len(data_3d_filepath) == 1:
        input_filepath = data_3d_filepath[0]
    else:
        raise ValueError(f"Expected 1 input files for '{data_3d_basename}' but got '{len(data_3d_filepath)}'.")

    # Pipeline Predicts
    predict_folder = PREDICT_PIPELINE_RESULTS_PATH
    predict_filepaths = sorted(pathlib.Path(predict_folder).rglob(f"{data_3d_basename}_*_output.*"))

    # Pipeline Merge output path
    output_folder = MERGE_PIPELINE_RESULTS_PATH
    os.makedirs(output_folder, exist_ok=True)

    # Start
    input_data = convert_data_file_to_numpy(data_filepath=input_filepath)

    first_column = log_data.columns[0]
    regex_pattern = f"{data_3d_basename}_.*"
    matching_rows = log_data[log_data[first_column].str.contains(regex_pattern, regex=True, na=False)]

    # Process the matching rows
    for row_idx, (_, row) in enumerate(matching_rows.iterrows()):
        predict_filepath = predict_filepaths[row_idx]
        predict_data = convert_data_file_to_numpy(data_filepath=predict_filepath)

        start_x, end_x, start_y, end_y, start_z, end_z = (
            row["start_x"], row["end_x"], row["start_y"], row["end_y"], row["start_z"], row["end_z"]
        )

        size_x, size_y, size_z = end_x - start_x, end_y - start_y, end_z - start_z

        # Perform the logical OR operation on the specific region
        input_data[start_x:end_x, start_y:end_y, start_z:end_z] = np.logical_or(
            input_data[start_x:end_x, start_y:end_y, start_z:end_z],
            predict_data[:size_x, :size_y, :size_z]
        )

    # Save the final result
    save_filename = os.path.join(output_folder, data_3d_basename)
    convert_numpy_to_data_file(numpy_data=input_data, source_data_filepath=input_filepath,
                               save_filename=save_filename)

    end_time = datetime.datetime.now()
    end_timestamp = end_time.strftime('%Y-%m-%d_%H-%M-%S')
    print(
        f"[Full Merge] Completed Merge... "
        f"(Timestamp: {end_timestamp}, Full Merge Time Elapsed: {end_time - start_time})"
    )


def calculate_dice_scores(data_3d_basename):
    #################
    # CROPS COMPARE #
    #################

    # Baseline
    # output_folder = PREDS_3D
    # output_filepaths = pathlib.Path(output_folder).rglob(f"{data_3d_basename}_*.*")

    # Output
    # output_folder = PREDICT_PIPELINE_RESULTS_PATH
    # output_filepaths = pathlib.Path(output_folder).rglob(f"{data_3d_basename}_*_output.*")

    # Ground Truth
    # target_folder = LABELS_3D
    # target_filepaths = pathlib.Path(target_folder).rglob(f"{data_3d_basename}_*.*")

    #####################
    # FULL DATA COMPARE #
    #####################

    # Baseline
    # output_folder = PREDS
    # output_filepaths = pathlib.Path(output_folder).rglob(f"{data_3d_basename}*.*")

    # Output
    output_folder = MERGE_PIPELINE_RESULTS_PATH
    output_filepaths = pathlib.Path(output_folder).rglob(f"{data_3d_basename}*.*")

    # Ground Truth
    target_folder = LABELS
    target_filepaths = pathlib.Path(target_folder).rglob(f"{data_3d_basename}*.*")

    #########################
    # Calculate Dice Scores #
    #########################

    output_filepaths = sorted(output_filepaths)
    target_filepaths = sorted(target_filepaths)

    filepaths_count = len(output_filepaths)
    scores_dict = dict()
    for idx in tqdm(range(filepaths_count)):
        output_filepath = output_filepaths[idx]
        target_filepath = target_filepaths[idx]

        output_3d_numpy = convert_data_file_to_numpy(data_filepath=output_filepath)
        target_3d_numpy = convert_data_file_to_numpy(data_filepath=target_filepath)

        dice_score = 2 * np.sum(output_3d_numpy * target_3d_numpy) / (np.sum(output_3d_numpy) + np.sum(target_3d_numpy))

        idx_format = get_data_file_stem(data_filepath=target_filepath)
        scores_dict[idx_format] = dice_score

    save_name = os.path.join(PREDICT_PIPELINE_RESULTS_PATH, "dice_scores.csv")
    pd.DataFrame(scores_dict.items()).to_csv(save_name)
    scores_list = list(scores_dict.values())
    print(
        "Stats:\n"
        f"Average Dice Score: {sum(scores_list) / len(scores_list)}\n"
        f"Max Dice Score: {max(scores_list)}\n"
        f"Min Dice Score: {min(scores_list)}"
    )


def full_folder_predict(data_3d_basename, data_type: DataType):
    raise NotImplementedError


def main():
    # 1. Use model 1 on the `parse_preds_mini_cropped`
    # 2. Save the results in `parse_fixed_mini_cropped`
    # 3. Perform direct `logical or` on `parse_fixed_mini_cropped` to get `parse_prefixed_mini_cropped_3d`
    # 4. Use model 2 on the `parse_prefixed_mini_cropped_3d`
    # 5. Save the results in `parse_fixed_mini_cropped_3d`
    # 6. Run steps 1-5 for mini cubes and combine all the results to get the final result
    # 7. Perform cleanup on the final result (delete small connected components)

    init_pipeline_models()

    # TODO: Requires Model Init
    # test_single_predict()

    data_3d_basename = "Hospital CUP 1in3"
    data_type = DataType.TRAIN
    full_predict(data_3d_basename=data_3d_basename, data_type=data_type)
    full_merge(data_3d_basename=data_3d_basename, data_type=data_type)
    # calculate_dice_scores(data_3d_basename=data_3d_basename)


if __name__ == "__main__":
    # RELEVANT TODOs
    # TODO: Add script for full folder predict

    # TODO: test loss functions to the 2D model

    # TODO: try to improve cleanup of 2D models results for the 3D fusion - TBD
    # TODO: support the classification models - TBD
    # TODO: add 45 degrees projections - Removed


    # TODO: another note for next stage:
    # 1. Add label on number of connected components inside a 3D volume
    # 2. Use the label to add a task for the model to predict the number of connected components


    parser = argparse.ArgumentParser(description='Main function to run the prediction pipeline')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='enables CUDA predicting')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--weights-filepath', type=str, default='./weights/Network.pth', metavar='N',
                        help='Which weights to use')
    parser.add_argument('--model-2d', type=str, default="", metavar='N',
                        help='Which 2D model to use')
    parser.add_argument('--input-size-model-2d', type=tuple, default=(1, 32, 32), metavar='N',
                        help='Which input size the 2D model should to use')
    parser.add_argument('--model-3d', type=str, default="", metavar='N',
                        help='Which 3D model to use')
    parser.add_argument('--input-size-model-3d', type=tuple, default=(1, 32, 32, 32), metavar='N',
                        help='Which input size the 3D model should to use')

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    torch.manual_seed(args.seed)

    # Custom Edit:

    # args.model_2d = "ae_6_2d_to_6_2d"
    # args.input_size_model_2d = (6, 32, 32)

    args.model_2d = "ae_2d_to_2d"
    args.input_size_model_2d = (1, DATA_2D_SIZE[0], DATA_2D_SIZE[1])

    # args.model_3d = "ae_3d_to_3d"
    args.model_3d = ""
    args.input_size_model_3d = (1, DATA_3D_SIZE[0], DATA_3D_SIZE[1], DATA_3D_SIZE[2])

    # TODO: Support different tasks

    main()
