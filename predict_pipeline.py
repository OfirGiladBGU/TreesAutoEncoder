import argparse
import os
import torch
import cv2
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import pathlib
from tqdm import tqdm
import pandas as pd
from scipy.ndimage import convolve

from datasets.dataset_utils import (get_data_file_stem, convert_data_file_to_numpy, convert_numpy_to_data_file,
                                    reverse_rotations, apply_threshold, IMAGES_6_VIEWS)
from datasets.dataset_list import DATASET_PATH, TRAIN_CROPPED_PATH, PREDICT_PIPELINE_RESULTS_PATH, MERGE_PIPELINE_RESULTS_PATH
from models.model_list import init_model


#########
# Utils #
#########
def preprocess_2d(data_3d_filepath, data_2d_folder, apply_batch_merge: bool = False) -> torch.Tensor:
    data_3d_basename = get_data_file_stem(data_filepath=data_3d_filepath)
    data_2d_basename = f"{data_3d_basename}_<VIEW>"

    # # Get relative path parts
    # relative_filepath = data_3d_filepath.relative_to(TRAIN_CROPPED_PATH)
    # relative_filepath_parts = list(relative_filepath.parts)
    #
    # # Update relative path parts to the relevant 2D images path
    # relative_filepath_parts[0] = relative_filepath_parts[0].replace("3d", "2d")
    # relative_filepath_parts[-1] = f"{data_2d_basename}.png"
    # format_of_2d_images_relative_filepath = pathlib.Path(*relative_filepath_parts)
    # format_of_2d_images = os.path.join(TRAIN_CROPPED_PATH, format_of_2d_images_relative_filepath)

    format_of_2d_images = os.path.join(data_2d_folder, data_2d_basename + ".png")

    # Projections 2D
    data_2d_list = list()
    for image_view in IMAGES_6_VIEWS:
        image_path = format_of_2d_images.replace("<VIEW>", image_view)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        torch_image = transforms.ToTensor()(image)
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


def postprocess_2d(data_2d_input: torch.Tensor, data_2d_output: torch.Tensor, apply_input_fusion: bool = False):
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

    if apply_input_fusion is True:
        data_2d_output = data_2d_input + torch.where(data_2d_input == 0, data_2d_output, 0)

    return data_2d_input, data_2d_output


def debug_2d(data_3d_filepath, data_2d_input: torch.Tensor, data_2d_output: torch.Tensor):
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


def noise_filter(data_3d_input: np.ndarray):
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


def preprocess_3d(data_3d_filepath,
                  data_2d_output: torch.Tensor,
                  apply_fusion: bool = False,
                  apply_noise_filter: bool = False):
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

    if apply_noise_filter is True:
        data_3d_input = noise_filter(data_3d_input=data_3d_input)

    data_3d_input = torch.Tensor(data_3d_input).unsqueeze(0).unsqueeze(0)
    return data_3d_input


def postprocess_3d(data_3d_input: torch.Tensor, data_3d_output: torch.Tensor, apply_input_fusion: bool = False):
    data_3d_input = data_3d_input.squeeze().squeeze()
    data_3d_output = data_3d_output.squeeze().squeeze()

    # TODO: Threshold
    apply_threshold(tensor=data_3d_output, threshold=0.5)

    if apply_input_fusion is True:
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
        model_2d_weights_filepath = f"{filepath}_{model_2d.model_name}{ext}"
        model_2d.load_state_dict(torch.load(model_2d_weights_filepath))
        model_2d.eval()
        model_2d.to(args.device)
        args.model_2d_class = model_2d

    # Load 3D model
    if len(args.model_3d) > 0:
        args.input_size = args.input_size_model_3d
        args.model = args.model_3d
        model_3d = init_model(args=args)
        model_3d_weights_filepath = f"{filepath}_{model_3d.model_name}{ext}"
        model_3d.load_state_dict(torch.load(model_3d_weights_filepath))
        model_3d.eval()
        model_3d.to(args.device)
        args.model_3d_class = model_3d


def single_predict(data_3d_filepath, data_2d_folder):
    data_3d_filepath = str(data_3d_filepath)
    data_2d_folder = str(data_2d_folder)

    os.makedirs(PREDICT_PIPELINE_RESULTS_PATH, exist_ok=True)

    if args.input_size_model_2d[0] == 6 and len(args.input_size_model_2d) == 3:
        apply_batch_merge = True
    else:
        apply_batch_merge = False

    with torch.no_grad():
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
        else:
            data_2d_output = data_2d_input.clone()

        # Parse 2D model output
        if "confidence map" in getattr(args.model_2d_class, "additional_tasks", list()):
            data_2d_output, data_2d_output_confidence = data_2d_output
            data_2d_output = torch.where(data_2d_output_confidence > 0.5, data_2d_output, 0)

        (data_2d_input, data_2d_output) = postprocess_2d(
            data_2d_input=data_2d_input,
            data_2d_output=data_2d_output,
            apply_input_fusion=True
        )

        # DEBUG
        debug_2d(data_3d_filepath=data_3d_filepath, data_2d_input=data_2d_input, data_2d_output=data_2d_output)

        ##############
        # 3D Section #
        ##############
        data_3d_input = preprocess_3d(
            data_3d_filepath=data_3d_filepath,
            data_2d_output=data_2d_output,
            apply_fusion=True,
            apply_noise_filter=True
        )

        # Predict 3D
        if len(args.model_3d) > 0:
            data_3d_output = args.model_3d_class(data_3d_input)
        else:
            data_3d_output = data_3d_input.clone()

        (data_3d_input, data_3d_output) = postprocess_3d(
            data_3d_input=data_3d_input,
            data_3d_output=data_3d_output,
            apply_input_fusion=True
        )

        # DEBUG
        debug_3d(data_3d_filepath=data_3d_filepath, data_3d_input=data_3d_input)

        export_output(data_3d_filepath=data_3d_filepath, data_3d_output=data_3d_output)


def test_single_predict():
    data_3d_filepath = os.path.join(TRAIN_CROPPED_PATH, "preds_3d_v6", "PA000005_11899.nii.gz")
    data_2d_folder = os.path.join(TRAIN_CROPPED_PATH, "preds_fixed_2d_v6")
    single_predict(
        data_3d_filepath=data_3d_filepath,
        data_2d_folder=data_2d_folder
    )


def full_predict():
    input_folder = os.path.join(TRAIN_CROPPED_PATH, "preds_3d_v6")
    data_2d_folder = os.path.join(TRAIN_CROPPED_PATH, "preds_fixed_2d_v6")

    # input_folder = os.path.join(CROPPED_PATH, "preds_fixed_3d_v6")

    input_format = "PA000005"
    data_3d_filepaths = pathlib.Path(input_folder).rglob(f"{input_format}_*.*")
    data_3d_filepaths = sorted(data_3d_filepaths)

    for data_3d_filepath in tqdm(data_3d_filepaths):
        single_predict(
            data_3d_filepath=data_3d_filepath,
            data_2d_folder=data_2d_folder
        )


def calculate_dice_scores():
    data_3d_basename = "PA000005"

    # output_folder = os.path.join(CROPPED_PATH, "preds_fixed_3d_v6")
    # output_filepaths = pathlib.Path(output_folder).rglob(f"{data_3d_basename}_*.*")

    output_folder = PREDICT_PIPELINE_RESULTS_PATH
    output_filepaths = pathlib.Path(output_folder).rglob(f"{data_3d_basename}_*_output.*")

    target_folder = os.path.join(TRAIN_CROPPED_PATH, "labels_3d_v6")
    target_filepaths = pathlib.Path(target_folder).rglob(f"{data_3d_basename}_*.*")

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


def full_merge():
    data_3d_basename = "PA000005"

    # Input 3D object
    input_folder = os.path.join(DATASET_PATH, "preds")
    input_filepath = list(pathlib.Path(input_folder).rglob(f"{data_3d_basename}*"))
    if len(input_filepath) == 1:
        input_filepath = input_filepath[0]
    else:
        raise ValueError(f"Expected 1 input files for '{data_3d_basename}' but got '{len(input_filepath)}'.")

    # Log file
    # TODO: create csv log per 3D object to improve search
    log_path = os.path.join(TRAIN_CROPPED_PATH, "log.csv")

    # Pipeline Predicts
    predict_folder = PREDICT_PIPELINE_RESULTS_PATH
    predict_filepaths = sorted(pathlib.Path(predict_folder).rglob(f"{data_3d_basename}_*_output.*"))

    # Pipeline Merge output path
    output_folder = MERGE_PIPELINE_RESULTS_PATH
    os.makedirs(output_folder, exist_ok=True)

    # Start
    input_data = convert_data_file_to_numpy(data_filepath=input_filepath)
    log_data = pd.read_csv(log_path)

    first_column = log_data.columns[0]
    regex_pattern = f"{data_3d_basename}_.*"
    matching_rows = log_data[log_data[first_column].str.contains(regex_pattern, regex=True, na=False)]

    # Process the matching rows
    for idx, row in matching_rows.iterrows():
        predict_filepath = predict_filepaths[idx]
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


def main():
    # 1. Use model 1 on the `parse_preds_mini_cropped_v5`
    # 2. Save the results in `parse_fixed_mini_cropped_v5`
    # 3. Perform direct `logical or` on `parse_fixed_mini_cropped_v5` to get `parse_prefixed_mini_cropped_3d_v5`
    # 4. Use model 2 on the `parse_prefixed_mini_cropped_3d_v5`
    # 5. Save the results in `parse_fixed_mini_cropped_3d_v5`
    # 6. Run steps 1-5 for mini cubes and combine all the results to get the final result
    # 7. Perform cleanup on the final result (delete small connected components)

    init_pipeline_models()

    # TODO: Requires Model Init
    # test_single_predict()
    full_predict()

    # calculate_dice_scores()
    full_merge()


if __name__ == "__main__":
    # 1
    # TODO: validate that after the 2d projection reconstruct, applying again 2d projection give the same results (artificial tests) - DONE
    # TODO: add plots of 3d data in matplotlib - DONE
    # TODO: creation of the 3d data of fixed 3d pred - DONE


    # 2
    # TODO: add 45 degrees projections
    # TODO: use the ground truth to create create a holes


    # 3
    # TODO: create the classification models


    # TODO: try to find new loss functions to the 2D model
    # TODO: try to check how to cleanup the 2D models results on the 3D fusion
    # TODO: find a way to organize the plots - DONE
    # TODO: test the full pipeline on the fixed preds

    # completed
    # TODO: run test of different inputs (100 examples)
    # TODO: show best 10, worst 10, median 10

    # TODO: another note for next stage:
    # 1. Add label on number of connected components inside a 3D volume
    # 2. Use the label to add a task for the model to predict the number of connected components
    parser = argparse.ArgumentParser(description='Main function to run the prediction pipeline')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='enables CUDA predicting')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--weights-filepath', type=str, default='./weights/Network.pth', metavar='N',
                        help='Which dataset to use')
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
    args.input_size_model_2d = (1, 32, 32)

    args.model_3d = "ae_3d_to_3d"
    args.input_size_model_3d = (1, 32, 32, 32)

    main()
