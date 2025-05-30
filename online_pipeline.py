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

from datasets_forge.dataset_2d_creator import crop_mini_cubes


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
            eval_component_filepath = None
            eval_components_cubes = None

        else:
            raise ValueError("Invalid Task Type")

        print(f"Total Mini Cubes: {len(eval_cubes)}\n")

        cubes_count = len(eval_cubes)
        cubes_count_digits_count = len(str(cubes_count))
        for cube_idx in tqdm(range(cubes_count)):
            # Get index data:
            eval_cube = eval_cubes[cube_idx]

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
                # TODO: Implement

            else:
                raise ValueError("Invalid Task Type")

            # Project 3D to 2D (Evals)
            eval_projections = project_3d_to_2d(
                data_3d=eval_cube,
                projection_options=projection_options,
                source_data_filepath=eval_filepath,
                component_3d=eval_components_cube
            )

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

            cube_idx_str = str(cube_idx).zfill(cubes_count_digits_count)

            output_3d_format = f"{output_idx}_{cube_idx_str}"
            log_data[output_3d_format] = cubes_data[cube_idx]
            projections_data[output_3d_format] = eval_projections

    log_data = pd.DataFrame(data=log_data).T
    return log_data, projections_data


def full_folder_predict(input_folder):
    data_3d_list = list(pathlib.Path(input_folder).rglob("*.*"))
    data_3d_count = len(data_3d_list)

    for idx, data_3d_filepath in enumerate(data_3d_list):
        data_3d_stem = get_data_file_stem(data_filepath=data_3d_filepath, relative_to=input_folder)
        print(f"[File: {data_3d_stem}, Number: {idx + 1}/{data_3d_count}] Predicting...")

        # Extracting log data
        log_data, projections_data = prepare_2d_projections_and_3d_cubes(
            input_filepath=data_3d_filepath,
            input_folder=input_folder
        )


def main():
    # TODO: Update as required
    input_folder = EVALS

    init_pipeline_models()

    # TODO: Create online full predict function
    full_folder_predict(input_folder)


if __name__ == "__main__":
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

    args.model_2d = "ae_2d_to_2d"
    args.input_size_model_2d = (1, DATA_2D_SIZE[0], DATA_2D_SIZE[1])

    # args.model_3d = "ae_3d_to_3d"
    args.model_3d = ""
    args.input_size_model_3d = (1, DATA_3D_SIZE[0], DATA_3D_SIZE[1], DATA_3D_SIZE[2])

    main()