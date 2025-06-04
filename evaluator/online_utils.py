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

from datasets_forge.dataset_2d_creator import crop_mini_cubes


def prepare_2d_projections_and_3d_cubes(input_filepath, input_folder):
    # Config
    projection_options = {
        "front": True,
        "back": True,
        "top": True,
        "bottom": True,
        "left": True,
        "right": True
    }

    # Inputs
    input_folders = {
        "evals": input_folder
    }
    # TODO: create online
    # if TASK_TYPE == TaskType.SINGLE_COMPONENT:
    #     input_folders.update({
    #         "evals_components": EVALS_COMPONENTS
    #     })

    # Log Data
    log_data = dict()
    projections_data = dict()

    # Get the filepaths
    input_filepaths = {
        "evals": [input_filepath]
    }
    # filepaths_found = list()
    # for key, value in input_folders.items():
    #     input_filepaths[key] = sorted(pathlib.Path(value).rglob("*.*"))
    #     filepaths_found.append(len(input_filepaths[key]))
    #
    # # Validation
    # if len(set(filepaths_found)) != 1:
    #     raise ValueError("Different number of files found in the Input folders")

    print("Cropping Mini Cubes...")
    filepaths_count = len(input_filepaths["evals"])
    # if STOP_INDEX > 0:
    #     filepaths_count = min(filepaths_count, STOP_INDEX)
    for filepath_idx in range(filepaths_count):
        # Get index data:
        eval_filepath = input_filepaths["evals"][filepath_idx]

        # Original data
        eval_numpy_data = convert_data_file_to_numpy(data_filepath=eval_filepath)

        # Crop Mini Cubes
        output_idx = get_data_file_stem(data_filepath=eval_filepath, relative_to=input_folders["evals"])
        print(f"[File: {output_idx}, Number: {filepath_idx + 1}/{filepaths_count}]")

        eval_cubes, cubes_data = crop_mini_cubes(
            data_3d=eval_numpy_data,
            cube_dim=DATA_3D_SIZE,
            stride_dim=DATA_3D_STRIDE,
            cubes_data=True,
            index_3d=f"~{output_idx}"
        )

        if TASK_TYPE == TaskType.SINGLE_COMPONENT:
            # TODO: create online
            # Get index data:
            # eval_component_filepath = input_filepaths["evals_components"][filepath_idx]

            # Original data
            # eval_component_numpy_data = convert_data_file_to_numpy(data_filepath=eval_component_filepath)

            (eval_component_numpy_data, _) = connected_components_3d(data_3d=eval_numpy_data)

            # Crop Mini Cubes
            eval_components_cubes = crop_mini_cubes(
                data_3d=eval_component_numpy_data,
                cube_dim=DATA_3D_SIZE,
                stride_dim=DATA_3D_STRIDE
            )

        elif TASK_TYPE in [TaskType.LOCAL_CONNECTIVITY, TaskType.PATCH_HOLES]:
            # eval_component_filepath = None
            eval_components_cubes = None

        else:
            raise ValueError("Invalid Task Type")

        # print(f"Total Mini Cubes: {len(eval_cubes)}\n")

        cubes_count = len(eval_cubes)
        cubes_count_digits_count = len(str(cubes_count))
        for cube_idx in tqdm(range(cubes_count)):
            # Get index data:
            eval_cube = eval_cubes[cube_idx]

            cube_idx_str = str(cube_idx).zfill(cubes_count_digits_count)

            # TODO: enable 2 modes
            if TASK_TYPE == TaskType.SINGLE_COMPONENT:
                # Get index data:
                eval_components_cube = eval_components_cubes[cube_idx]

                # TASK CONDITION: The region has 2 or more different global components
                global_components_3d_indices = list(np.unique(eval_components_cube))
                global_components_3d_indices.remove(0)
                global_components_3d_count = len(global_components_3d_indices)
                if global_components_3d_count < 2:
                    continue

                # Log 3D info
                cubes_data[cube_idx].update({
                    # "name": output_3d_format,
                    "eval_global_components": global_components_3d_count,
                })

            elif TASK_TYPE == TaskType.LOCAL_CONNECTIVITY:
                eval_components_cube = None

                # TASK CONDITION: NONE

            elif TASK_TYPE == TaskType.PATCH_HOLES:
                eval_components_cube = None

                # TASK CONDITION: The region has holes

            else:
                raise ValueError("Invalid Task Type")

            # Project 3D to 2D (Evals)
            eval_projections = project_3d_to_2d(
                data_3d=eval_cube,
                projection_options=projection_options,
                source_data_filepath=eval_filepath,
                component_3d=eval_components_cube
            )

            ########################
            # Density Filter Crops #
            ########################

            condition_list = [True] * len(IMAGES_6_VIEWS)
            for view_idx, image_view in enumerate(IMAGES_6_VIEWS):
                eval_image = eval_projections[f"{image_view}_image"]

                condition = [
                    not (UPPER_THRESHOLD_2D > np.count_nonzero(eval_image) > LOWER_THRESHOLD_2D)
                ]

                # Check Condition (If condition fails, skip the current view):
                if any(condition):
                    condition_list[view_idx] = False

                    cubes_data[cube_idx].update({
                        f"{image_view}_valid": False
                    })
                else:
                    cubes_data[cube_idx].update({
                        f"{image_view}_valid": True
                    })

            # Validate that at least 1 condition is met (if not, pop cube data)
            if not any(condition_list):
                continue

            ###########################
            # Export Data - In Memory #
            ###########################

            output_3d_format = f"{output_idx}_{cube_idx_str}"
            log_data[output_3d_format] = cubes_data[cube_idx]

            eval_projections["cube"] = eval_cube
            projections_data[output_3d_format] = eval_projections

    log_data = pd.DataFrame(data=log_data).T
    return log_data, projections_data


def online_preprocess_2d(data_3d_stem, projections_data, apply_batch_merge=False):
    # Projections 2D
    data_2d_list = list()
    for image_view in IMAGES_6_VIEWS:
        image_data = projections_data[data_3d_stem][f"{image_view}_image"]
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


def online_preprocess_3d(data_3d_filepath: str,
                         projections_data: dict,
                         data_2d_output: torch.Tensor = None,
                         apply_threshold_2d: bool = False,
                         threshold_2d: float = 0.2,
                         apply_fusion: bool = False,
                         apply_noise_filter_3d: bool = False,
                         hard_noise_filter_3d: bool = True,
                         connectivity_type_3d: int = 6) -> torch.Tensor:
    data_3d_stem = get_data_file_stem(data_filepath=data_3d_filepath)
    pred_3d = projections_data[data_3d_stem]["cube"]

    # 2D flow was disabled
    if data_2d_output is None:
        data_3d_input = pred_3d

    # Use 2D flow results
    else:
        data_2d_output = data_2d_output.numpy()

        # TODO: Threshold
        if apply_threshold_2d:
            apply_threshold(data_2d_output, threshold=threshold_2d, keep_values=True)

        # Reconstruct 3D
        data_3d_list = list()
        for idx, image_view in enumerate(IMAGES_6_VIEWS):
            numpy_image = data_2d_output[idx] * 255
            data_3d = reverse_rotations(numpy_image=numpy_image, view_type=image_view,
                                        source_data_filepath=data_3d_filepath)
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