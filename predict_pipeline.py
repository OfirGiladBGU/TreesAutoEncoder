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
from statistics import mean

from datasets_forge.dataset_configurations import *
from datasets.dataset_utils import *
from models.model_list import init_model
# TODO: Debug Tools
from datasets_visualize.dataset_visulalization import interactive_plot_2d, interactive_plot_3d

#########
# Utils #
#########
def preprocess_2d(data_3d_stem: str,
                  data_2d_folder: str,
                  apply_batch_merge: bool = False) -> torch.Tensor:
    data_2d_stem = f"{data_3d_stem}_<VIEW>"

    # # Get relative path parts
    # relative_filepath = data_3d_filepath.relative_to(CROPS_PATH)
    # relative_filepath_parts = list(relative_filepath.parts)
    #
    # # Update relative path parts to the relevant 2D images path
    # relative_filepath_parts[0] = relative_filepath_parts[0].replace("3d", "2d")
    # relative_filepath_parts[-1] = f"{data_2d_stem}.png"
    # format_of_2d_images_relative_filepath = pathlib.Path(*relative_filepath_parts)
    # format_of_2d_images = os.path.join(CROPS_PATH, format_of_2d_images_relative_filepath)

    format_of_2d_images = os.path.join(data_2d_folder, f"{data_2d_stem}.png")

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


def postprocess_2d(data_3d_stem: str,
                   data_2d_input: torch.Tensor,
                   data_2d_output: torch.Tensor,
                   apply_input_merge_2d: bool = False,
                   apply_noise_filter_2d: bool = False,
                   hard_noise_filter_2d: bool = True,
                   connectivity_type_2d: int = 4,
                   log_data: pd.DataFrame = None) -> Tuple[torch.Tensor, torch.Tensor]:
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

    if log_data is not None:
        # Replace invalid data with original input
        for idx, image_view in enumerate(IMAGES_6_VIEWS):
            view_advance_valid = f"{image_view}_advance_valid"
            if view_advance_valid in log_data.columns:
                data_2d_matching_rows = log_data[log_data[log_data.columns[0]] == data_3d_stem]
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

    if apply_input_merge_2d is True:
        data_2d_output = data_2d_input + torch.where(data_2d_input == 0, data_2d_output, 0)

    if apply_noise_filter_2d is True:
        for idx in range(len(IMAGES_6_VIEWS)):
            data_2d_input_idx = np.round(data_2d_input[idx].numpy() * 255).astype(np.uint8)
            data_2d_output_idx = np.round(data_2d_output[idx].numpy() * 255).astype(np.uint8)

            # if TASK_TYPE == TaskType.SINGLE_COMPONENT:
            #     filtered_output = components_continuity_2d_single_component(
            #         label_image=data_2d_input_idx,
            #         pred_advanced_fixed_image=data_2d_output_idx,
            #         reverse_mode=False,
            #         binary_diff=True,
            #         hard_condition=hard_noise_filter_2d
            #     )
            # elif TASK_TYPE == TaskType.LOCAL_CONNECTIVITY:
            #     filtered_output = components_continuity_2d_local_connectivity(
            #         label_image=data_2d_input_idx,
            #         pred_advanced_fixed_image=data_2d_output_idx,
            #         reverse_mode=False,
            #         binary_diff=True,
            #         hard_condition=hard_noise_filter_2d
            #     )
            # else:
            #     filtered_output = data_2d_output_idx

            filtered_output = components_continuity_2d_local_connectivity(
                label_image=data_2d_input_idx,
                pred_advanced_fixed_image=data_2d_output_idx,
                reverse_mode=False,
                connectivity_type=connectivity_type_2d,
                binary_diff=True,
                hard_condition=hard_noise_filter_2d
            )

            data_2d_output[idx] = torch.Tensor(filtered_output / 255.0)

    return data_2d_input, data_2d_output


def debug_2d(data_3d_stem: str, data_2d_input: torch.Tensor, data_2d_output: torch.Tensor):
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
        numpy_image = np.round(numpy_image * 255).astype(np.uint8)
        numpy_image = np.expand_dims(numpy_image, axis=-1)
        plt.imshow(numpy_image, cmap='gray')
        ax[j].set_title(f"View {IMAGES_6_VIEWS[j]}:")

    # 2D Output
    for j in range(columns):
        ax.append(fig.add_subplot(rows, columns, 1 * columns + j + 1))
        numpy_image = data_2d_output_copy[j]
        numpy_image = np.round(numpy_image * 255).astype(np.uint8)
        numpy_image = np.expand_dims(numpy_image, axis=-1)
        plt.imshow(numpy_image, cmap='gray')
        ax[j].set_title(f"View {IMAGES_6_VIEWS[j]}:")

    save_path = os.path.join(PREDICT_PIPELINE_RESULTS_PATH, "output_2d_debug")
    save_filename = os.path.join(save_path, data_3d_stem)
    os.makedirs(name=os.path.dirname(save_filename), exist_ok=True)
    fig.tight_layout()
    plt.savefig(save_filename)
    plt.close(fig)


def export_output_2d(data_3d_stem, data_3d_filepath, data_2d_output: torch.Tensor):
    numpy_data_2d_output = data_2d_output.clone().numpy()

    source_data_filepath = "dummy.png"
    save_path = os.path.join(PREDICT_PIPELINE_RESULTS_PATH, "output_2d")
    for view_idx, image_view in enumerate(IMAGES_6_VIEWS):
        numpy_image = numpy_data_2d_output[view_idx]
        numpy_image = np.round(numpy_image * 255).astype(np.uint8)

        save_filename = os.path.join(save_path, f"{data_3d_stem}_{image_view}_output")
        convert_numpy_to_data_file(
            numpy_data=numpy_image,
            source_data_filepath=source_data_filepath,
            save_filename=save_filename,
        )


def naive_noise_filter(data_3d_original: np.ndarray, data_3d_input: np.ndarray):
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


def preprocess_3d(data_3d_filepath: str,
                  data_2d_output: torch.Tensor = None,
                  apply_threshold_2d: bool = False,
                  apply_fusion: bool = False,
                  apply_noise_filter_3d: bool = False,
                  hard_noise_filter_3d: bool = True,
                  connectivity_type_3d: int = 6) -> torch.Tensor:
    pred_3d = convert_data_file_to_numpy(data_filepath=data_3d_filepath, apply_data_threshold=True)

    # 2D flow was disabled
    if data_2d_output is None:
        data_3d_input = pred_3d

    # Use 2D flow results
    else:
        data_2d_output = data_2d_output.numpy()

        # TODO: Threshold
        if apply_threshold_2d:
            apply_threshold(data_2d_output, threshold=0.2, keep_values=True)

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
        apply_threshold(data_3d_reconstruct, threshold=0.5, keep_values=False)

        # Fusion 3D
        if apply_fusion is True:
            data_3d_fusion = np.logical_or(data_3d_reconstruct, pred_3d)
            data_3d_fusion = data_3d_fusion.astype(np.float32)
            apply_threshold(data_3d_fusion, threshold=0.5, keep_values=False)
            data_3d_input = data_3d_fusion
        else:
            data_3d_input = data_3d_reconstruct

        # TODO: validate filters!!!
        if apply_noise_filter_3d is True:
            # data_3d_input = naive_noise_filter(data_3d_original=pred_3d, data_3d_input=data_3d_input)

            # if TASK_TYPE == TaskType.SINGLE_COMPONENT:
            #     data_3d_input = components_continuity_3d_single_component(
            #         label_cube=pred_3d,
            #         pred_advanced_fixed_cube=data_3d_input,
            #         reverse_mode=False,
            #         connectivity_type=connectivity_type_3d,
            #         hard_condition=hard_noise_filter_3d
            #     )
            # elif TASK_TYPE == TaskType.LOCAL_CONNECTIVITY:
            #     data_3d_input = components_continuity_3d_local_connectivity(
            #         label_cube=pred_3d,
            #         pred_advanced_fixed_cube=data_3d_input,
            #         reverse_mode=False,
            #         connectivity_type=connectivity_type_3d,
            #         hard_condition=hard_noise_filter_3d
            #     )
            # else:
            #     pass

            data_3d_input = components_continuity_3d_local_connectivity(
                label_cube=pred_3d,
                pred_advanced_fixed_cube=data_3d_input,
                reverse_mode=False,
                connectivity_type=connectivity_type_3d,
                hard_condition=hard_noise_filter_3d
            )

    data_3d_input = torch.Tensor(data_3d_input).unsqueeze(0).unsqueeze(0)
    return data_3d_input


def postprocess_3d(data_3d_input: torch.Tensor,
                   data_3d_output: torch.Tensor,
                   apply_input_merge_3d: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    data_3d_input = data_3d_input.squeeze().squeeze()
    data_3d_output = data_3d_output.squeeze().squeeze()

    # TODO: Threshold
    apply_threshold(data_3d_output, threshold=0.5, keep_values=False)

    if apply_input_merge_3d is True:
        data_3d_output = data_3d_input + torch.where(data_3d_input == 0, data_3d_output, 0)

    return data_3d_input, data_3d_output


def debug_3d(data_3d_stem, data_3d_filepath, data_3d_input: torch.Tensor):
    data_3d_input = data_3d_input.clone().numpy()

    save_path = os.path.join(PREDICT_PIPELINE_RESULTS_PATH, "output_3d_debug")
    save_filename = os.path.join(save_path, f"{data_3d_stem}_input")
    convert_numpy_to_data_file(
        numpy_data=data_3d_input,
        source_data_filepath=data_3d_filepath,
        save_filename=save_filename,
        apply_data_threshold=True
    )


def export_output_3d(data_3d_stem, data_3d_filepath, data_3d_output: torch.Tensor):
    numpy_data_3d_output = data_3d_output.numpy()

    save_path = os.path.join(PREDICT_PIPELINE_RESULTS_PATH, "output_3d")
    save_filename = os.path.join(save_path, f"{data_3d_stem}_output")
    convert_numpy_to_data_file(
        numpy_data=numpy_data_3d_output,
        source_data_filepath=data_3d_filepath,
        save_filename=save_filename,
        apply_data_threshold=True
    )


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
        # TODO: model_2d_weights_filepath = f"{filepath}_PipeForge3DPCD_Best_LC_32_{model_2d.model_name}{ext}"
        # TODO: model_2d_weights_filepath = f"{filepath}_PipeForge3DMesh_Base_LC_32_{model_2d.model_name}{ext}"
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
        # TODO: model_3d_weights_filepath = f"{filepath}_PipeForge3DPCD_LC_32_{model_3d.model_name}{ext}"
        model_3d.load_state_dict(torch.load(model_3d_weights_filepath))
        model_3d.eval()
        model_3d.to(args.device)
        args.model_3d_class = model_3d
    else:
        pass


def single_predict(data_3d_filepath, data_3d_folder, data_2d_folder,
                   log_data=None, enable_debug=True,
                   run_2d_flow=True, run_3d_flow=True,
                   export_2d=True, export_3d=True):
    if not os.path.exists(data_3d_filepath):
        raise ValueError(f"File: {data_3d_filepath} doesn't exist!")

    # CONFIGS
    apply_input_merge_2d = False  # False - Doesn't work well with revealed occluded objects
    apply_input_merge_3d = True
    apply_threshold_2d = True
    apply_fusion = True

    apply_noise_filter_2d = False  # Notice: Doesn't work well with revealed occluded objects
    hard_noise_filter_2d = True
    connectivity_type_2d = 4

    apply_noise_filter_3d = True
    hard_noise_filter_3d = True
    connectivity_type_3d = 6

    # INPUTS
    data_3d_filepath = str(data_3d_filepath)
    data_3d_folder = str(data_3d_folder)
    data_2d_folder = str(data_2d_folder)

    data_3d_stem = get_data_file_stem(data_filepath=data_3d_filepath, relative_to=data_3d_folder)

    # os.makedirs(PREDICT_PIPELINE_RESULTS_PATH, exist_ok=True)

    if args.input_size_model_2d[0] == 6 and len(args.input_size_model_2d) == 3:
        apply_batch_merge = True
    else:
        apply_batch_merge = False

    with torch.no_grad():
        # TODO: Support 1D model TBD

        ##############
        # 2D Section #
        ##############

        if run_2d_flow is True:
            # Preprocess 2D
            data_2d_input = preprocess_2d(
                data_3d_stem=data_3d_stem,
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

            # Postprocess 2D
            (data_2d_input, data_2d_output) = postprocess_2d(
                data_3d_stem=data_3d_stem,
                data_2d_input=data_2d_input,
                data_2d_output=data_2d_output,
                apply_input_merge_2d=apply_input_merge_2d,
                apply_noise_filter_2d=apply_noise_filter_2d,
                hard_noise_filter_2d=hard_noise_filter_2d,
                connectivity_type_2d=connectivity_type_2d,
                log_data=log_data
            )

            # DEBUG
            if enable_debug is True:
                debug_2d(data_3d_stem=data_3d_stem, data_2d_input=data_2d_input, data_2d_output=data_2d_output)

            if export_2d is True:
                export_output_2d(
                    data_3d_stem=data_3d_stem,
                    data_3d_filepath=data_3d_filepath,
                    data_2d_output=data_2d_output
                )
        else:
            data_2d_output = None

        ##############
        # 3D Section #
        ##############

        if run_3d_flow is True:
            # Preprocess 3D
            data_3d_input = preprocess_3d(
                data_3d_filepath=data_3d_filepath,
                data_2d_output=data_2d_output,
                apply_threshold_2d=apply_threshold_2d,
                apply_fusion=apply_fusion,
                apply_noise_filter_3d=apply_noise_filter_3d,
                hard_noise_filter_3d=hard_noise_filter_3d,
                connectivity_type_3d=connectivity_type_3d
            )

            # Predict 3D
            if len(args.model_3d) > 0:
                data_3d_output = args.model_3d_class(data_3d_input)
            else:
                data_3d_output = data_3d_input.clone()

            # Postprocess 3D
            (data_3d_input, data_3d_output) = postprocess_3d(
                data_3d_input=data_3d_input,
                data_3d_output=data_3d_output,
                apply_input_merge_3d=apply_input_merge_3d
            )

            # DEBUG
            if enable_debug is True:
                debug_3d(data_3d_stem=data_3d_stem, data_3d_filepath=data_3d_filepath, data_3d_input=data_3d_input)
                # TODO: Add matplotlib export

            if export_3d is True:
                export_output_3d(
                    data_3d_stem=data_3d_stem,
                    data_3d_filepath=data_3d_filepath,
                    data_3d_output=data_3d_output
                )
        else:
            pass


def test_single_predict():
    data_type = DataType.TRAIN
    # data_type = DataType.EVAL

    # base_filename = "PA000005_11899.nii.gz"
    # base_filename = "PA000078_11996.nii.gz"
    # base_filename = "47_52.npy"
    base_filename = "46_301.npy"

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
        data_2d_folder = EVALS_2D

    # Get filepaths
    data_3d_filepath = os.path.join(data_3d_folder, base_filename)
    single_predict(
        data_3d_filepath=data_3d_filepath,
        data_3d_folder=data_3d_folder,
        data_2d_folder=data_2d_folder,
        log_data=log_data
    )


def full_predict(data_3d_stem, data_type: DataType, log_data=None, data_3d_folder=None, data_2d_folder=None,
                 run_2d_flow=True, run_3d_flow=True, export_2d=True, export_3d=True):
    # LOAD DATA #
    if data_type == DataType.TRAIN:
        if log_data is None:
            log_data = pd.read_csv(TRAIN_LOG_PATH)

        if data_3d_folder is None:
            # data_3d_folder = PREDS_3D
            data_3d_folder = PREDS_FIXED_3D
            # data_3d_folder = PREDS_ADVANCED_FIXED_3D

        if data_2d_folder is None:
            # data_2d_folder = PREDS_2D
            data_2d_folder = PREDS_FIXED_2D
            # data_2d_folder = PREDS_ADVANCED_FIXED_2D

    else:
        if log_data is None:
            log_data = pd.read_csv(EVAL_LOG_PATH)

        if data_3d_folder is None:
            data_3d_folder = EVALS_3D

        if data_2d_folder is None:
            data_2d_folder = EVALS_2D

    # Get filepaths (based on folder)
    # data_3d_filepaths = pathlib.Path(data_3d_folder).glob(f"{data_3d_stem}_*.*")
    # data_3d_filepaths = sorted(data_3d_filepaths)

    # Get filepath (based on csv)
    data_3d_filepaths = []
    col_0 = log_data.columns[0]
    for row_idx, row in log_data.iterrows():
        # Skip non relevant rows
        if data_3d_stem != str(row[col_0]).rsplit("_", maxsplit=1)[0]:
            continue

        data_3d_file = list(pathlib.Path(data_3d_folder).glob(f"{row[col_0]}.*"))[0]
        data_3d_filepaths.append(data_3d_file)

    # START #
    start_time = datetime.datetime.now()
    start_timestamp = start_time.strftime('%Y-%m-%d_%H-%M-%S')
    print(
        f"[Full Predict] Started Predict... "
        f"(Timestamp: {start_timestamp})"
    )

    # Single-threading - Sequential
    # for data_3d_filepath in tqdm(data_3d_filepaths):
    #     single_predict(
    #         data_3d_filepath=data_3d_filepath,
    #         data_3d_folder=data_3d_folder,
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
                    data_3d_folder=data_3d_folder,
                    data_2d_folder=data_2d_folder,
                    log_data=log_data,
                    enable_debug=False,
                    run_2d_flow=run_2d_flow,
                    run_3d_flow=run_3d_flow,
                    export_2d=export_2d,
                    export_3d=export_3d
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


def full_merge(data_3d_stem, data_type: DataType, log_data=None, source_data_3d_folder=None):
    # LOAD DATA #
    if data_type == DataType.TRAIN:
        if log_data is None:
            log_data = pd.read_csv(TRAIN_LOG_PATH)

        if source_data_3d_folder is None:
            source_data_3d_folder = PREDS
            # source_data_3d_folder = PREDS_FIXED
            # source_data_3d_folder = PREDS_ADVANCED_FIXED

    else:
        if log_data is None:
            log_data = pd.read_csv(EVAL_LOG_PATH)

        if source_data_3d_folder is None:
            source_data_3d_folder = EVALS

    # Input 3D object
    data_3d_filepath = list(pathlib.Path(source_data_3d_folder).glob(f"{data_3d_stem}*"))
    if len(data_3d_filepath) == 1:
        input_filepath = data_3d_filepath[0]
    else:
        raise ValueError(f"Expected 1 input files for '{data_3d_stem}' but got '{len(data_3d_filepath)}'.")

    # Pipeline Predicts
    predict_folder = PREDICT_PIPELINE_RESULTS_PATH
    predict_filepaths = sorted(pathlib.Path(os.path.join(predict_folder, "output_3d")).glob(f"{data_3d_stem}_*_output.*"))

    # Pipeline Merge output path
    output_folder = MERGE_PIPELINE_RESULTS_PATH

    # os.makedirs(output_folder, exist_ok=True)

    input_data = convert_data_file_to_numpy(data_filepath=input_filepath, apply_data_threshold=True)

    first_column = log_data.columns[0]
    regex_pattern = f"{data_3d_stem}_.*"
    matching_rows = log_data[log_data[first_column].str.contains(regex_pattern, regex=True, na=False)]

    # START #
    start_time = datetime.datetime.now()
    start_timestamp = start_time.strftime('%Y-%m-%d_%H-%M-%S')
    print(
        f"[Full Merge] Started Merge... "
        f"(Timestamp: {start_timestamp})"
    )

    # Process the matching rows
    for row_idx, (_, row) in enumerate(matching_rows.iterrows()):
        predict_filepath = predict_filepaths[row_idx]
        predict_data = convert_data_file_to_numpy(data_filepath=predict_filepath, apply_data_threshold=True)

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
    save_filename = os.path.join(output_folder, data_3d_stem)
    convert_numpy_to_data_file(
        numpy_data=input_data,
        source_data_filepath=input_filepath,
        save_filename=save_filename,
        apply_data_threshold=True
    )

    end_time = datetime.datetime.now()
    end_timestamp = end_time.strftime('%Y-%m-%d_%H-%M-%S')
    print(
        f"[Full Merge] Completed Merge... "
        f"(Timestamp: {end_timestamp}, Full Merge Time Elapsed: {end_time - start_time})"
    )


def calculate_2d_avg_l1_error(data_3d_stem, data_2d_folder=None):
    # Baseline
    # input_folder = PREDS_2D
    # output_folder = PREDS_FIXED_2D
    # input_folder = PREDS_ADVANCED_FIXED_2D
    # output_filepaths = list(pathlib.Path(output_folder).glob(f"{data_3d_stem}*.*"))

    # Output
    output_folder = os.path.join(PREDICT_PIPELINE_RESULTS_PATH, "output_2d")
    output_filepaths = list(pathlib.Path(output_folder).glob(f"{data_3d_stem}_*_*_output.*"))

    # Input
    if data_2d_folder is None:
        # input_folder = PREDS_2D
        input_folder = PREDS_FIXED_2D
        # input_folder = PREDS_ADVANCED_FIXED_2D
    else:
        input_folder = data_2d_folder

    input_filepaths = []
    for output_filepath in output_filepaths:
        output_filepath_extension = get_data_file_extension(data_filepath=output_filepath)
        output_filepath_stem = get_data_file_stem(data_filepath=output_filepath, relative_to=output_folder)
        if output_folder == os.path.join(PREDICT_PIPELINE_RESULTS_PATH, "output_2d"):
            output_filepath_stem = str(output_filepath_stem).rsplit('_output', maxsplit=1)[0]
        output_filename = f"{output_filepath_stem}{output_filepath_extension}"

        input_filepath = os.path.join(input_folder, output_filename)
        input_filepaths.append(input_filepath)

    # Ground Truth
    target_folder = LABELS_2D
    target_filepaths = []
    for output_filepath in output_filepaths:
        output_filepath_extension = get_data_file_extension(data_filepath=output_filepath)
        output_filepath_stem = get_data_file_stem(data_filepath=output_filepath, relative_to=output_folder)
        if output_folder == os.path.join(PREDICT_PIPELINE_RESULTS_PATH, "output_2d"):
            output_filepath_stem = str(output_filepath_stem).rsplit('_output', maxsplit=1)[0]
        output_filename = f"{output_filepath_stem}{output_filepath_extension}"

        target_filepath = os.path.join(target_folder, output_filename)
        target_filepaths.append(target_filepath)

    l1_avg_errors_list = []
    l1_max_errors_list = []

    total_hole_pixels = 0
    total_detected_pixels = 0
    total_close_pixels = 0
    total_fill_pixels = 0

    for idx in range(len(output_filepaths)):
        input_filepath_idx = input_filepaths[idx]
        target_filepath_idx = target_filepaths[idx]
        output_filepath_idx = output_filepaths[idx]

        input_data = convert_data_file_to_numpy(data_filepath=input_filepath_idx)
        target_data = convert_data_file_to_numpy(data_filepath=target_filepath_idx)
        output_data = convert_data_file_to_numpy(data_filepath=output_filepath_idx)

        delta_binary_mask = (np.abs(target_data - input_data) > 0.5)
        if delta_binary_mask.sum() == 0:
            continue

        masked_input = input_data[delta_binary_mask]
        masked_output = output_data[delta_binary_mask]
        masked_target = target_data[delta_binary_mask]

        l1_error = np.abs(masked_target - masked_output)

        avg_l1_error = np.mean(l1_error)
        l1_avg_errors_list.append(avg_l1_error)
        max_l1_error = np.max(l1_error)
        l1_max_errors_list.append(max_l1_error)

        # Assumption: Filled holes should have higher pixel value than unfilled holes
        condition_matrix1 = (masked_output > masked_input)  # Increase in color
        condition_matrix2 = (np.abs(masked_target - masked_output) < (255 * 0.25))  # Error is less than 25%
        condition_matrix3 = np.logical_and(condition_matrix1, condition_matrix2)  # Increase in color & error is less than 25%

        total_detected_pixels += np.sum(condition_matrix1)
        total_close_pixels += np.sum(condition_matrix2)
        total_fill_pixels += np.sum(condition_matrix3)

        total_hole_pixels += np.sum(delta_binary_mask)

    if len(l1_avg_errors_list) > 0:
        final_avg_l1_error = np.mean(np.array(l1_avg_errors_list))
        final_max_l1_error = np.max(np.array(l1_max_errors_list))
        detected_percentage = total_detected_pixels / total_hole_pixels
        close_percentage = total_close_pixels / total_hole_pixels
        fill_percentage = total_fill_pixels / total_hole_pixels
        # fill_percentage = total_fill_pixels / total_detected_pixels

        print(
            "Stats:\n"
            f"Hole Average L1 Error: {final_avg_l1_error}\n"
            f"Hole Max L1 Error: {final_max_l1_error}\n"
            f"Increase in hole pixel value % (Detect Rate): {detected_percentage}\n"
            f"Less than 25% error on hole pixel value % (Close Rate): {close_percentage}\n"
            f"Detect and Close Rates combined % (Fill Rate): {fill_percentage}"
        )

        output_dict = {
            "Detect Rate": detected_percentage,
            "Close Rate": close_percentage,
            "Fill Pixels": fill_percentage,
        }
        return output_dict
    else:
        print("No L1 Errors found")


def calculate_dice_scores(data_3d_stem, compare_crops_mode: bool = False):
    #################
    # CROPS COMPARE #
    #################
    if compare_crops_mode is True:
        # Baseline
        # output_folder = PREDS_3D
        # output_folder = PREDS__FIXED_3D
        # output_filepaths = list(pathlib.Path(output_folder).glob(f"{data_3d_stem}_*.*"))

        # Output
        output_folder = os.path.join(PREDICT_PIPELINE_RESULTS_PATH, "output_3d")
        output_filepaths = list(pathlib.Path(output_folder).glob(f"{data_3d_stem}_*_output.*"))

        # Ground Truth
        target_folder = LABELS_3D
        # target_filepaths = list(pathlib.Path(target_folder).glob(f"{data_3d_stem}_*.*"))
        target_filepaths = []
        for output_filepath in output_filepaths:
            output_filepath_extension = get_data_file_extension(data_filepath=output_filepath)
            output_filepath_stem = get_data_file_stem(data_filepath=output_filepath, relative_to=output_folder)
            if output_folder == os.path.join(PREDICT_PIPELINE_RESULTS_PATH, "output_3d"):
                output_filepath_stem = str(output_filepath_stem).rsplit('_output', maxsplit=1)[0]
            output_filename = f"{output_filepath_stem}{output_filepath_extension}"

            target_filepath = os.path.join(target_folder, output_filename)
            target_filepaths.append(target_filepath)

    #####################
    # FULL DATA COMPARE #
    #####################
    else:
        # Notice: A list of 1 file will be compared

        # Baseline
        # output_folder = PREDS
        # output_folder = PREDS_FIXED
        # output_filepaths = list(pathlib.Path(output_folder).glob(f"{data_3d_stem}*.*"))

        # Output
        output_folder = MERGE_PIPELINE_RESULTS_PATH
        output_filepaths = list(pathlib.Path(output_folder).glob(f"{data_3d_stem}*.*"))

        # Ground Truth
        target_folder = LABELS
        target_filepaths = list(pathlib.Path(target_folder).glob(f"{data_3d_stem}*.*"))

        # TODO: FOR SAFETY
        # target_filepaths = []
        # for output_filepath in output_filepaths:
        #     output_filepath_extension = get_data_file_extension(data_filepath=output_filepath)
        #     output_filepath_stem = get_data_file_stem(data_filepath=output_filepath, relative_to=output_folder)
        #     output_filename = f"{output_filepath_stem}{output_filepath_extension}"
        #
        #     target_filepath = os.path.join(target_folder, output_filename)
        #     target_filepaths.append(target_filepath)

    #########################
    # Calculate Dice Scores #
    #########################

    # output_filepaths = sorted(output_filepaths)
    # target_filepaths = sorted(target_filepaths)

    filepaths_count = len(output_filepaths)
    scores_dict = dict()
    for idx in tqdm(range(filepaths_count)):
        output_filepath = output_filepaths[idx]
        target_filepath = target_filepaths[idx]

        output_3d_numpy = convert_data_file_to_numpy(data_filepath=output_filepath, apply_data_threshold=True)
        target_3d_numpy = convert_data_file_to_numpy(data_filepath=target_filepath, apply_data_threshold=True)

        dice_score = 2 * np.sum(output_3d_numpy * target_3d_numpy) / (np.sum(output_3d_numpy) + np.sum(target_3d_numpy))

        idx_format = get_data_file_stem(data_filepath=target_filepath)
        scores_dict[idx_format] = dice_score

    save_filepath = os.path.join(PREDICT_PIPELINE_DICE_CSV_FILES_PATH, f"{data_3d_stem}_dice_scores.csv")
    os.makedirs(name=os.path.dirname(save_filepath), exist_ok=True)
    pd.DataFrame(scores_dict.items()).to_csv(save_filepath)
    scores_list = list(scores_dict.values())
    avg_dice =  sum(scores_list) / len(scores_list)
    print(
        "Stats:\n"
        f"Average Dice Score: {avg_dice}\n"
        f"Max Dice Score: {max(scores_list)}\n"
        f"Min Dice Score: {min(scores_list)}"
    )

    output_dict = {
        "Dice Score": avg_dice,
    }
    return output_dict


def calculate_reduced_connected_components(data_3d_stem, components_mode="global", source_data_3d_folder=None):
    """
    Requires data_3d_stem result in MERGE_PIPELINE_RESULTS_PATH
    Works only for SINGLE_COMPONENT mode
    :param data_3d_stem:
    :param components_mode:
    :param source_data_3d_folder:
    :return:
    """

    # Baseline
    if source_data_3d_folder is None:
        # input_folder = PREDS
        input_folder = PREDS_FIXED
    else:
        input_folder = source_data_3d_folder
    input_filepath = list(pathlib.Path(input_folder).glob(f"{data_3d_stem}*.*"))[0]

    # Output
    output_folder = MERGE_PIPELINE_RESULTS_PATH
    output_filepath = list(pathlib.Path(output_folder).glob(f"{data_3d_stem}*.*"))[0]

    # Ground Truth
    target_folder = LABELS
    target_filepath = list(pathlib.Path(target_folder).glob(f"{data_3d_stem}*.*"))[0]

    #############
    # Load Data #
    #############
    input_data_3d = convert_data_file_to_numpy(data_filepath=input_filepath, apply_data_threshold=True)
    output_data_3d = convert_data_file_to_numpy(data_filepath=output_filepath, apply_data_threshold=True)
    target_data_3d = convert_data_file_to_numpy(data_filepath=target_filepath, apply_data_threshold=True)

    # Calculate connected components
    if components_mode == "global":
        connectivity_type = 26

        # Notice: Usually `target_connected_components=1`
        (_, input_connected_components) = connected_components_3d(data_3d=input_data_3d, connectivity_type=connectivity_type)
        (_, output_connected_components) = connected_components_3d(data_3d=output_data_3d, connectivity_type=connectivity_type)
        (_, target_connected_components) = connected_components_3d(data_3d=target_data_3d, connectivity_type=connectivity_type)

        # Formula: (Total Components Reduced / Total Components needs to be Reduced)
        reduction_percentage = ((input_connected_components - output_connected_components) /
                                (input_connected_components - target_connected_components))
        print(
            "Stats:\n"
            f"Input Connected Components: {input_connected_components}\n"
            f"Output Connected Components: {output_connected_components}\n"
            f"Target Connected Components: {target_connected_components}\n"
            f"Reduction Percentage: {reduction_percentage}"
        )

    elif components_mode == "local":
        # # Option 1 - Check for each output fill, if some connections where closed
        #
        # apply_dilation_scope = True
        # connectivity_type = 26
        # padding_size = 1
        #
        # padded_input = pad_data(numpy_data=input_data_3d, pad_width=padding_size)
        # padded_output = pad_data(numpy_data=output_data_3d, pad_width=padding_size)
        # # padded_target = pad_data(numpy_data=target_data_3d, pad_width=padding_size)
        #
        # delta_binary = np.logical_xor(padded_output, padded_input).astype(np.int16)
        # delta_labeled, delta_num_components = connected_components_3d(
        #     data_3d=delta_binary,
        #     connectivity_type=connectivity_type
        # )
        #
        # filled_holes = 0
        # total_holes = delta_num_components
        #
        # if total_holes == 0:
        #     print(
        #         "Stats:\n"
        #         f"Input Holes: 0\n"
        #         f"Output Filled Holes: 0\n"
        #         f"Reduction Percentage: 1.0"
        #     )
        #
        # print(f"Total Holes Found: {total_holes}. Checking Filled Holes...")
        #
        # # Iterate through connected components in delta_cube
        # expand_mask = None
        # for component_label in tqdm(range(1, delta_num_components + 1)):
        #     # Create a mask for the current connected component
        #     component_mask = np.equal(delta_labeled, component_label).astype(np.uint8)
        #
        #     # ROI - cropped area between the component mask: top, bottom, left, right, front, back
        #     coords = np.argwhere(component_mask > 0)
        #
        #     # Get bounding box in 3D
        #     top = np.min(coords[:, 0])  # Minimum row index (Y-axis)
        #     bottom = np.max(coords[:, 0])  # Maximum row index (Y-axis)
        #
        #     left = np.min(coords[:, 1])  # Minimum column index (X-axis)
        #     right = np.max(coords[:, 1])  # Maximum column index (X-axis)
        #
        #     front = np.min(coords[:, 2])  # Minimum depth index (Z-axis)
        #     back = np.max(coords[:, 2])  # Maximum depth index (Z-axis)
        #
        #     # Ensure ROI is within valid bounds
        #     min_y = top - padding_size
        #     max_y = bottom + padding_size + 1
        #
        #     min_x = left - padding_size
        #     max_x = right + padding_size + 1
        #
        #     min_z = front - padding_size
        #     max_z = back + padding_size + 1
        #
        #     # Check the number of connected components before adding the mask
        #     roi_temp_before = padded_input[min_y:max_y, min_x:max_x, min_z:max_z]
        #     if apply_dilation_scope is True:
        #         expand_mask = get_local_scope_mask(numpy_data=roi_temp_before, padding_size=padding_size)
        #         apply_local_scope_mask(numpy_data=roi_temp_before, expand_mask=expand_mask)
        #     (_, components_before) = connected_components_3d(data_3d=roi_temp_before, connectivity_type=connectivity_type)
        #
        #     # Create a temporary data with the component added
        #     temp_fix = np.logical_or(padded_input, component_mask)
        #     roi_temp_after = temp_fix[min_y:max_y, min_x:max_x, min_z:max_z]
        #     if apply_dilation_scope is True:
        #         apply_local_scope_mask(numpy_data=roi_temp_after, expand_mask=expand_mask)
        #     (_, components_after) = connected_components_3d(data_3d=roi_temp_after, connectivity_type=connectivity_type)
        #
        #     # if components_after < components_before:
        #     if components_after <= components_before:
        #         filled_holes += 1
        #
        # reduction_percentage = filled_holes / total_holes
        #
        # print(
        #     "Stats:\n"
        #     f"Input Holes: {total_holes}\n"
        #     f"Output Filled Holes: {filled_holes}\n"
        #     f"Reduction Percentage: {reduction_percentage}"
        # )
        #
        # Option 2 - Compare how many hole components left after removing the output fills (Works only on fully completed holes)
        #
        # connectivity_type = 26
        #
        # input_delta_data_3d = ((target_data_3d - input_data_3d) > 0.5).astype(np.uint8)
        # (_, input_delta_num_components) = connected_components_3d(
        #     data_3d=input_delta_data_3d,
        #     connectivity_type=connectivity_type
        # )
        #
        # output_delta_data_3d = ((target_data_3d - output_data_3d) > 0.5).astype(np.uint8)
        # (_, output_delta_num_components) = connected_components_3d(
        #     data_3d=output_delta_data_3d,
        #     connectivity_type=connectivity_type
        # )
        #
        # reduction_percentage = (input_delta_num_components - output_delta_num_components) / input_delta_num_components
        #
        # print(
        #     "Stats:\n"
        #     f"Input Holes: {input_delta_num_components}\n"
        #     f"Output Holes: {output_delta_num_components}\n"
        #     f"Reduction Percentage: {reduction_percentage}"
        # )

        # Option 3 - Comparing if there was reduction in the true holes locations (High Runtime)

        # mode 1 - compare if Output components is smaller than Input components
        # mode 2 - compare GT and Output components on the locals scope
        # compare_mode = 1
        #
        # apply_dilation_scope = True
        # connectivity_type = 26
        # padding_size = 1
        #
        # padded_input = pad_data(numpy_data=input_data_3d, pad_width=padding_size)
        # padded_output = pad_data(numpy_data=output_data_3d, pad_width=padding_size)
        # padded_target = pad_data(numpy_data=target_data_3d, pad_width=padding_size)
        #
        # delta_binary = np.logical_xor(padded_target, padded_input).astype(np.int16)
        # delta_labeled, delta_num_components = connected_components_3d(
        #     data_3d=delta_binary,
        #     connectivity_type=connectivity_type
        # )
        #
        # filled_holes = 0
        # total_holes = delta_num_components
        #
        # if total_holes == 0:
        #     print(
        #         "Stats:\n"
        #         f"Input Holes: 0\n"
        #         f"Output Filled Holes: 0\n"
        #         f"Reduction Percentage: 1.0"
        #     )
        #
        # print(f"Total Holes Found: {total_holes}. Checking Filled Holes...")
        #
        # # Iterate through connected components in delta_cube
        # expand_mask = None
        # for component_label in tqdm(range(1, delta_num_components + 1)):
        #     # Create a mask for the current connected component
        #     component_mask = np.equal(delta_labeled, component_label).astype(np.uint8)
        #
        #     # ROI - cropped area between the component mask: top, bottom, left, right, front, back
        #     coords = np.argwhere(component_mask > 0)
        #
        #     # Get bounding box in 3D
        #     top = np.min(coords[:, 0])  # Minimum row index (Y-axis)
        #     bottom = np.max(coords[:, 0])  # Maximum row index (Y-axis)
        #
        #     left = np.min(coords[:, 1])  # Minimum column index (X-axis)
        #     right = np.max(coords[:, 1])  # Maximum column index (X-axis)
        #
        #     front = np.min(coords[:, 2])  # Minimum depth index (Z-axis)
        #     back = np.max(coords[:, 2])  # Maximum depth index (Z-axis)
        #
        #     # Ensure ROI is within valid bounds
        #     min_y = top - padding_size
        #     max_y = bottom + padding_size + 1
        #
        #     min_x = left - padding_size
        #     max_x = right + padding_size + 1
        #
        #     min_z = front - padding_size
        #     max_z = back + padding_size + 1
        #
        #     # Check the number of connected components before adding the mask
        #     if compare_mode == 1:
        #         roi_temp_before = padded_input[min_y:max_y, min_x:max_x, min_z:max_z]
        #     else:
        #         roi_temp_before = padded_target[min_y:max_y, min_x:max_x, min_z:max_z]
        #     if apply_dilation_scope is True:
        #         expand_mask = get_local_scope_mask(numpy_data=roi_temp_before, padding_size=padding_size)
        #         apply_local_scope_mask(numpy_data=roi_temp_before, expand_mask=expand_mask)
        #     (_, components_before) = connected_components_3d(data_3d=roi_temp_before,
        #                                                      connectivity_type=connectivity_type)
        #
        #     # Create a temporary data with the component added
        #     roi_temp_after = padded_output[min_y:max_y, min_x:max_x, min_z:max_z]
        #     if apply_dilation_scope is True:
        #         apply_local_scope_mask(numpy_data=roi_temp_after, expand_mask=expand_mask)
        #     (_, components_after) = connected_components_3d(data_3d=roi_temp_after, connectivity_type=connectivity_type)
        #
        #     if compare_mode == 1:
        #         condition = (components_after < components_before)
        #     else:
        #         condition = (components_after == components_before)
        #
        #     if condition:
        #         filled_holes += 1
        #     else:
        #         # print("DEBUG")
        #         pass
        #
        # reduction_percentage = filled_holes / total_holes
        #
        # print(
        #     "Stats:\n"
        #     f"Input Holes: {total_holes}\n"
        #     f"Output Filled Holes: {filled_holes}\n"
        #     f"Reduction Percentage: {reduction_percentage}"
        # )

        # Option 4 - Check coverage percentage of the holes

        connectivity_type = 26

        # delta_binary = ((target_data_3d - input_data_3d) > 0.5).astype(np.int16)
        delta_binary = np.logical_xor(target_data_3d, input_data_3d).astype(np.int16)
        (_, input_delta_num_components) = connected_components_3d(
            data_3d=delta_binary,
            connectivity_type=connectivity_type
        )
        delta_mask = delta_binary.astype(bool)

        target_holes = target_data_3d[delta_mask]
        output_holes = output_data_3d[delta_mask]
        dice_score = 2 * np.sum(output_holes * target_holes) / (np.sum(output_holes) + np.sum(target_holes))

        print(
            "Stats:\n"
            f"Input Holes: {input_delta_num_components}\n"
            f"Coverage Percentage: {dice_score}"
        )

    else:
        raise ValueError("Invalid components_mode")


def full_folder_predict(data_type: DataType):
    run_full_predict = True

    # Select which tests to run
    test_2d_avg_l1_error = True
    test_dice_score = True
    test_components_reduced = False

    ################
    # Prepare Data #
    ################

    if data_type == DataType.TRAIN:
        log_data = pd.read_csv(TRAIN_LOG_PATH)

        # source_data_3d_folder = PREDS
        source_data_3d_folder = PREDS_FIXED

        # data_3d_folder = LABELS_3D  # SANITY
        # data_3d_folder = PREDS_3D
        data_3d_folder = PREDS_FIXED_3D
        # data_3d_folder = PREDS_ADVANCED_FIXED_3D

        # data_2d_folder = LABELS_2D  # SANITY
        # data_2d_folder = PREDS_2D
        data_2d_folder = PREDS_FIXED_2D
        # data_2d_folder = PREDS_ADVANCED_FIXED_2D
    else:
        log_data = pd.read_csv(EVAL_LOG_PATH)

        source_data_3d_folder = EVALS

        data_3d_folder = EVALS_3D

        data_2d_folder = EVALS_2D

    index_3d_uniques = log_data["index_3d"].unique()
    data_3d_stem_list = [data_3d_stem[1:] for data_3d_stem in index_3d_uniques]

    # Evaluate on Training - Test Data
    if data_3d_folder != EVALS_3D:
        split_percentage = 0.9
        index_3d_split_index = min(round(len(index_3d_uniques) * split_percentage), len(index_3d_uniques) - 1)

        # train_stems = data_3d_stem_list[:index_3d_split_index]
        test_stems = data_3d_stem_list[index_3d_split_index:]

        data_3d_stem_list = test_stems

    data_3d_stem_count = len(data_3d_stem_list)

    ########################
    # Test 2D Avg L1 Error #
    ########################

    outputs = {
        "Detect Rate": [],
        "Close Rate": [],
        "Fill Pixels": [],
    }
    if test_2d_avg_l1_error:
        # TODO: Run the 2D models and compare the 2D results with the 2D GT

        for idx, data_3d_stem in enumerate(data_3d_stem_list):
            print(f"[File: {data_3d_stem}, Number: {idx + 1}/{data_3d_stem_count}] Predicting...")
            full_predict(
                data_3d_stem=data_3d_stem,
                data_type=data_type,
                log_data=log_data,
                data_3d_folder=data_3d_folder,
                data_2d_folder=data_2d_folder,
                run_3d_flow=False,
                export_3d=False
            )

        for idx, data_3d_stem in enumerate(data_3d_stem_list):
            print(f"[File: {data_3d_stem}, Number: {idx + 1}/{data_3d_stem_count}] Calculating 2D AVG L1 Error...")
            output = calculate_2d_avg_l1_error(data_3d_stem=data_3d_stem, data_2d_folder=data_2d_folder)

            for key in output.keys():
                outputs[key].append(output[key])

        output_str = "\n[AVG RESULTS]\n"
        for key in outputs.keys():
            output_str += f"AVG {key}: {mean(outputs[key])}\n"
        print(output_str)

    ###################
    # Test Dice Score #
    ###################

    if test_dice_score:
        compare_crops_mode = True

        if run_full_predict:
            for idx, data_3d_stem in enumerate(data_3d_stem_list):
                print(f"[File: {data_3d_stem}, Number: {idx + 1}/{data_3d_stem_count}] Predicting...")
                full_predict(
                    data_3d_stem=data_3d_stem,
                    data_type=data_type,
                    log_data=log_data,
                    data_3d_folder=data_3d_folder,
                    data_2d_folder=data_2d_folder,
                    export_2d=False
                )
            run_full_predict = False

        if compare_crops_mode is False:
            for idx, data_3d_stem in enumerate(data_3d_stem_list):
                print(f"[File: {data_3d_stem}, Number: {idx + 1}/{data_3d_stem_count}] Merging...")
                full_merge(
                    data_3d_stem=data_3d_stem,
                    data_type=data_type,
                    log_data=log_data,
                    source_data_3d_folder=source_data_3d_folder
                )

        outputs = {
            "Dice Score": []
        }
        for idx, data_3d_stem in enumerate(data_3d_stem_list):
            print(f"[File: {data_3d_stem}, Number: {idx + 1}/{data_3d_stem_count}] Calculating Dice Scores...")
            output = calculate_dice_scores(data_3d_stem=data_3d_stem, compare_crops_mode=compare_crops_mode)

            for key in output.keys():
                outputs[key].append(output[key])

        output_str = "\n[AVG RESULTS]\n"
        for key in outputs.keys():
            output_str += f"AVG {key}: {mean(outputs[key])}\n"
        print(output_str)

    ###########################
    # Test Components Reduced #
    ###########################

    if test_components_reduced:
        if run_full_predict:
            for idx, data_3d_stem in enumerate(data_3d_stem_list):
                print(f"[File: {data_3d_stem}, Number: {idx + 1}/{data_3d_stem_count}] Predicting...")
                full_predict(
                    data_3d_stem=data_3d_stem,
                    data_type=data_type,
                    log_data=log_data,
                    data_3d_folder=data_3d_folder,
                    data_2d_folder=data_2d_folder,
                    export_2d=False
                )
            run_full_predict = False

        for idx, data_3d_stem in enumerate(data_3d_stem_list):
            print(f"[File: {data_3d_stem}, Number: {idx + 1}/{data_3d_stem_count}] Merging...")
            full_merge(
                data_3d_stem=data_3d_stem,
                data_type=data_type,
                log_data=log_data,
                source_data_3d_folder=source_data_3d_folder
            )

        # if TASK_TYPE == TaskType.SINGLE_COMPONENT:
        #     components_mode = "global"
        # else:
        #     components_mode = "local"

        components_mode = "local"
        for idx, data_3d_stem in enumerate(data_3d_stem_list):
            print(f"[File: {data_3d_stem}, Number: {idx + 1}/{data_3d_stem_count}] Calculating Components Scores...")
            calculate_reduced_connected_components(
                data_3d_stem=data_3d_stem,
                components_mode=components_mode,
                source_data_3d_folder=source_data_3d_folder
            )


def main():
    init_pipeline_models()

    # TODO: Requires Model Init
    # test_single_predict()

    data_type = DataType.TRAIN
    # data_3d_stem = "PA000005"
    # data_3d_stem = "Hospital CUP 1in3"

    # full_predict(data_3d_stem=data_3d_stem, data_type=data_type, export_2d=False)
    # full_merge(data_3d_stem=data_3d_stem, data_type=data_type)
    # calculate_dice_scores(data_3d_stem=data_3d_stem)

    full_folder_predict(data_type=data_type)


if __name__ == "__main__":
    # RELEVANT TODOs
    # TODO: Add script for full folder predict

    # TODO: test loss functions to the 2D model

    # TODO: try to improve cleanup of 2D models results for the 3D fusion - TBD
    # TODO: support the classification models - TBD
    # TODO: add 45 degrees projections - Removed
    # TODO: create csv log per 3D object to improve search (Maybe in the future) - Removed


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
