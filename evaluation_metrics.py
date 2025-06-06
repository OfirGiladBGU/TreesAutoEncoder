import argparse
import os
import pathlib
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import datetime
from statistics import mean
from skimage.metrics import structural_similarity as ssim

from configs.configs_parser import *
from datasets.dataset_utils import *
from evaluator.predict_pipeline import init_pipeline_models, single_predict, full_merge

# TODO: Debug Tools
from datasets_visualize.dataset_visulalization import interactive_plot_2d, interactive_plot_3d


##################
# Core Functions #
##################
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
    data_3d_cube_filepaths = []
    col_0 = log_data.columns[0]
    for row_idx, row in log_data.iterrows():
        # Skip non relevant rows
        if data_3d_stem != str(row[col_0]).rsplit("_", maxsplit=1)[0]:
            continue

        data_3d_cube_filepath = list(pathlib.Path(data_3d_folder).glob(f"{row[col_0]}.*"))[0]
        data_3d_cube_filepaths.append(data_3d_cube_filepath)

    # START #
    start_time = datetime.datetime.now()
    start_timestamp = start_time.strftime('%Y-%m-%d_%H-%M-%S')
    print(
        f"[Full Predict] Started Predict... "
        f"(Timestamp: {start_timestamp})"
    )

    # Single-threading - Sequential
    # for data_3d_cube_filepath in tqdm(data_3d_cube_filepaths):
    #     single_predict(
    #         data_3d_filepath=data_3d_cube_filepath,
    #         data_3d_folder=data_3d_folder,
    #         data_2d_folder=data_2d_folder,
    #         log_data=log_data,
    #         enable_debug=True
    #     )

    # Multi-threading
    futures = []
    with ThreadPoolExecutor() as executor:
        # Submit all tasks
        for data_3d_cube_filepath in data_3d_cube_filepaths:
            futures.append(
                executor.submit(
                    single_predict,
                    args=args,
                    data_3d_filepath=data_3d_cube_filepath,
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


def compute_depth_completion_metrics(output, target, mask, epsilon=1e-6):
    # Apply mask
    masked_output = output[mask]
    masked_target = target[mask]

    # Basic metrics
    mae = np.mean(np.abs(masked_output - masked_target))
    rmse = np.sqrt(np.mean((masked_output - masked_target) ** 2))
    rel = np.mean(np.abs(masked_output - masked_target) / (masked_target + epsilon))  # avoid division by 0

    # Threshold accuracies
    thresh = np.maximum(masked_output / (masked_target + epsilon), masked_target / (masked_output + epsilon))
    delta1 = np.mean(thresh < 1.25)
    delta2 = np.mean(thresh < 1.25 ** 2)
    delta3 = np.mean(thresh < 1.25 ** 3)
    ssim_score = ssim(output, target, data_range=target.max() - target.min())

    results = {
        "Masked MAE": mae,
        "Masked RMSE": rmse,
        "Masked REL": rel,
        "Masked Delta1": delta1,
        "Masked Delta2": delta2,
        "Masked Delta3": delta3,
        "SSIM": ssim_score
    }
    return results


def calculate_2d_metrics(data_3d_stem, data_2d_folder=None, apply_abs: bool = True):
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

    results_list = {}
    for idx in range(len(output_filepaths)):
        input_filepath_idx = input_filepaths[idx]
        target_filepath_idx = target_filepaths[idx]
        output_filepath_idx = output_filepaths[idx]

        input_data = convert_data_file_to_numpy(data_filepath=input_filepath_idx)
        target_data = convert_data_file_to_numpy(data_filepath=target_filepath_idx)
        output_data = convert_data_file_to_numpy(data_filepath=output_filepath_idx)

        if apply_abs:
            delta_binary_mask = (np.abs(target_data - input_data) > 0.5)
        else:
            delta_binary_mask = (target_data - input_data) > 0.5
        if delta_binary_mask.sum() == 0:
            continue

        results = compute_depth_completion_metrics(
            output=output_data,
            target=target_data,
            mask=delta_binary_mask
        )

        for key, value in results.items():
            if key in results_list:
                results_list[key].append(value)
            else:
                results_list[key] = [value]

    # Calculate average results
    output_dict = {}
    for key in results_list.keys():
        output_dict[key] = mean(results_list[key]) if results_list[key] else 0

    # Print results
    print_str = "[AVG RESULTS]\n"
    for key, value in output_dict.items():
        print_str += f"AVG {key}: {value}\n"
    print(print_str)
    return output_dict


def calculate_2d_custom_metrics(data_3d_stem, data_2d_folder=None, apply_abs: bool = True):
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

        if apply_abs:
            delta_binary_mask = (np.abs(target_data - input_data) > 0.5)
        else:
            delta_binary_mask = (target_data - input_data) > 0.5
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
            f"Detect and Close Rates combined % (Fill Rate): {fill_percentage}\n"
        )

        output_dict = {
            "Detect Rate": detected_percentage,
            "Close Rate": close_percentage,
            "Fill Pixels": fill_percentage
        }
        return output_dict
    else:
        print("No L1 Errors found")
        output_dict = {
            "Detect Rate": 1.0,
            "Close Rate": 1.0,
            "Fill Pixels": 1.0
        }
        return output_dict


def calculate_dice_scores(data_3d_stem, compare_crops_mode: bool = False):
    #################
    # CROPS COMPARE #
    #################
    if compare_crops_mode is True:
        # Baseline
        # output_folder = PREDS_3D
        # output_folder = PREDS_FIXED_3D
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

        # Patch for Parse2022 (Medpseg)
        # target_filepaths = list(pathlib.Path(target_folder).glob(f"{data_3d_stem.replace('_vessel', '')}*.*"))

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
        f"Min Dice Score: {min(scores_list)}\n"
    )

    output_dict = {
        "Dice Score": avg_dice,
    }
    return output_dict


def calculate_reduced_connected_components(data_3d_stem, components_mode="global", source_data_3d_folder=None,
                                           apply_abs: bool = True):
    """
    Requires data_3d_stem result in MERGE_PIPELINE_RESULTS_PATH
    Works only for SINGLE_COMPONENT mode
    :param data_3d_stem:
    :param components_mode:
    :param source_data_3d_folder:
    :param apply_abs
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
            f"Reduction Percentage: {reduction_percentage}\n"
        )

    elif components_mode == "local":
        connectivity_type = 26

        if apply_abs:
            # delta_binary = (abs(target_data_3d - input_data_3d) > 0.5).astype(np.int16)
            delta_binary = np.logical_xor(target_data_3d, input_data_3d).astype(np.int16)
        else:
            delta_binary = ((target_data_3d - input_data_3d) > 0.5).astype(np.int16)

        (_, input_delta_num_components) = connected_components_3d(
            data_3d=delta_binary,
            connectivity_type=connectivity_type
        )
        delta_mask = delta_binary.astype(bool)

        if delta_mask.sum() > 0:
            target_holes = target_data_3d[delta_mask]
            output_holes = output_data_3d[delta_mask]
            dice_score = 2 * np.sum(output_holes * target_holes) / (np.sum(output_holes) + np.sum(target_holes))
        else:
            dice_score = 1.0

        print(
            "Stats:\n"
            f"Input Holes: {input_delta_num_components}\n"
            f"Coverage Percentage: {dice_score}\n"
        )

    else:
        raise ValueError("Invalid components_mode")


def full_folder_predict(data_type: DataType):
    run_full_predict = True
    compare_crops_mode = True
    apply_abs = True  # True - When the input has no outliers (Train Data), False - When the input has outliers (Eval Data)

    # Select which tests to run
    test_2d_metrics = True
    test_2d_custom_metrics = False
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

    ###################
    # Test 2D Metrics #
    ###################
    outputs = {}
    if test_2d_metrics:
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
            print(f"[File: {data_3d_stem}, Number: {idx + 1}/{data_3d_stem_count}] Calculating 2D Metrics...")
            output = calculate_2d_metrics(
                data_3d_stem=data_3d_stem,
                data_2d_folder=data_2d_folder,
                apply_abs=apply_abs
            )

            for key, value in output.items():
                if key in outputs:
                    outputs[key].append(value)
                else:
                    outputs[key] = [value]

        output_str = "[AVG RESULTS]\n"
        for key in outputs.keys():
            output_str += f"AVG {key}: {mean(outputs[key])}\n"
        print(output_str)

    ##########################
    # Test 2D Custom Metrics #
    ##########################

    outputs = {
        "Detect Rate": [],
        "Close Rate": [],
        "Fill Pixels": [],
    }
    if test_2d_custom_metrics:
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
            output = calculate_2d_custom_metrics(
                data_3d_stem=data_3d_stem,
                data_2d_folder=data_2d_folder,
                apply_abs=apply_abs
            )

            for key in output.keys():
                outputs[key].append(output[key])

        output_str = "[AVG RESULTS]\n"
        for key in outputs.keys():
            output_str += f"AVG {key}: {mean(outputs[key])}\n"
        print(output_str)

    ###################
    # Test Dice Score #
    ###################

    if test_dice_score:
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

        output_str = "[AVG RESULTS]\n"
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
                source_data_3d_folder=source_data_3d_folder,
                apply_abs=apply_abs
            )


def main():
    init_pipeline_models(args=args)

    # TODO: Requires Model Init
    data_type = DataType.TRAIN
    full_folder_predict(data_type=data_type)


if __name__ == "__main__":
    # RELEVANT TODOs
    # TODO: Add script for full folder predict

    # TODO: test loss functions to the 2D model

    # TODO: try to improve cleanup of 2D models results for the 3D fusion - TBD
    # TODO: support the classification models - TBD
    # TODO: add 45 degrees projections - Removed
    # TODO: create csv log per 3D object to improve search (Maybe in the future) - Removed

    # TODO: Support different tasks

    # TODO: another note for next stage:
    # 1. Add label on number of connected components inside a 3D volume
    # 2. Use the label to add a task for the model to predict the number of connected components

    parser = argparse.ArgumentParser(description='Main function to run the prediction pipeline')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='enables CUDA predicting')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 1)')
    # parser.add_argument('--weights-filepath', type=str, default='./weights/Network.pth', metavar='N',
    #                     help='Which weights to use')  # Moved to YAML config
    parser.add_argument('--model-1d', type=str, default="", metavar='N',
                        help='Which 1D model to use')
    parser.add_argument('--input-size-model-1d', type=tuple, default=(1, 32, 32), metavar='N',
                        help='Which input size the 2D model should to use')
    parser.add_argument('--model-2d', type=str, default="", metavar='N',
                        help='Which 2D model to use')
    parser.add_argument('--input-size-model-2d', type=tuple, default=(1, 32, 32), metavar='N',
                        help='Which input size the 2D model should to use')
    parser.add_argument('--model-3d', type=str, default="", metavar='N',
                        help='Which 3D model to use')
    parser.add_argument('--input-size-model-3d', type=tuple, default=(1, 32, 32, 32), metavar='N',
                        help='Which input size the 3D model should to use')

    args = parser.parse_args()
    args.mode = "offline"

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    torch.manual_seed(args.seed)

    # Custom Edit:

    # args.model_1d = "vit_2d_to_1d"
    # args.input_size_model_2d = (1, DATA_2D_SIZE[0], DATA_2D_SIZE[1])

    # args.model_2d = "ae_6_2d_to_6_2d"
    # args.input_size_model_2d = (6, DATA_2D_SIZE[0], DATA_2D_SIZE[1])

    args.model_2d = "ae_2d_to_2d"
    args.input_size_model_2d = (1, DATA_2D_SIZE[0], DATA_2D_SIZE[1])

    # args.model_3d = "ae_3d_to_3d"
    # args.input_size_model_3d = (1, DATA_3D_SIZE[0], DATA_3D_SIZE[1], DATA_3D_SIZE[2])

    main()
