import argparse
import os
import torch
import cv2
import numpy as np
import nibabel as nib
from torchvision import transforms
import matplotlib.pyplot as plt
import pathlib
from tqdm import tqdm
import pandas as pd

from datasets.dataset_utils import convert_nii_gz_to_numpy, convert_numpy_to_nii_gz, reverse_rotations, apply_threshold
from datasets.dataset_list import CROPPED_PATH
from models.ae_2d_to_2d import Network
from models.ae_3d_to_3d import Network3D


# TODO: split 2d and 3d pre pre processes to different function

def single_predict(format_of_2d_images, output_path):
    os.makedirs(output_path, exist_ok=True)

    # Load models
    filepath, ext = os.path.splitext(args.weights_filepath)

    args.input_size = (1, 32, 32)
    model_2d = Network(args=args)
    model_2d_weights_filepath = f"{filepath}_{model_2d.model_name}{ext}"
    model_2d.load_state_dict(torch.load(model_2d_weights_filepath))
    model_2d.eval()

    args.input_size = (1, 32, 32, 32)
    model_3d = Network3D(args=args)
    model_3d_weights_filepath = f"{filepath}_{model_3d.model_name}{ext}"
    model_3d.load_state_dict(torch.load(model_3d_weights_filepath))
    model_3d.eval()

    with torch.no_grad():
        ###################
        # Prepare 2D data #
        ###################

        # Projections 2D
        images_6_views = ['top', 'bottom', 'front', 'back', 'left', 'right']
        data_2d_list = list()
        for image_view in images_6_views:
            image_path = format_of_2d_images.replace("<VIEW>", image_view)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            torch_image = transforms.ToTensor()(image)
            data_2d_list.append(torch_image)
        data_2d = torch.stack(data_2d_list)

        # Predict 2D
        data_2d_predicts = model_2d(data_2d)

        ###################
        # Prepare 3D data #
        ###################

        # Reconstruct 3D
        data_2d_predicts = data_2d_predicts.numpy()
        data_2d = data_2d.numpy()

        # TODO: DEBUG - START
        columns = 6
        rows = 2
        fig = plt.figure(figsize=(columns + 0.5, rows + 0.5))
        ax = []
        for j in range(columns):
            ax.append(fig.add_subplot(rows, columns, 0 * columns + j + 1))
            npimg = data_2d[j]
            npimg = npimg * 255
            npimg = npimg.astype(np.uint8)
            plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')

            ax[j].set_title(f"View {images_6_views[j]}:")


        for j in range(columns):
            ax.append(fig.add_subplot(rows, columns, 1 * columns + j + 1))
            npimg = data_2d_predicts[j]
            npimg = npimg * 255
            npimg = npimg.astype(np.uint8)
            plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')

            ax[j].set_title(f"View {images_6_views[j]}:")

        fig.tight_layout()
        plt.savefig(os.path.join("predict_results", f"input_output_images.png"))
        plt.close(fig)
        # TODO: DEBUG - END

        # Reconstruct 3D
        data_3d_list = list()
        for idx, image_view in enumerate(images_6_views):
            numpy_image = data_2d_predicts[idx].squeeze() * 255
            data_3d = reverse_rotations(numpy_image=numpy_image, view_type=image_view)
            data_3d_list.append(data_3d)

        data_3d_reconstruct = data_3d_list[0]
        for i in range(1, len(data_3d_list)):
            data_3d_reconstruct = np.logical_or(data_3d_reconstruct, data_3d_list[i])
        data_3d_reconstruct = data_3d_reconstruct.astype(np.float32)

        # Fusion 3D
        pred_3d_filepath = os.path.join(CROPPED_PATH, "preds_3d_v6", os.path.basename(format_of_2d_images).replace("_<VIEW>.png", ".nii.gz"))
        pred_3d = convert_nii_gz_to_numpy(data_filepath=pred_3d_filepath)
        data_3d_fusion = np.logical_or(data_3d_reconstruct, pred_3d)

        # TODO: DEBUG - START
        save_name = os.path.join(output_path, os.path.basename(format_of_2d_images).replace("_<VIEW>.png", "_input"))
        convert_numpy_to_nii_gz(numpy_data=data_3d_fusion, save_name=save_name)
        # TODO: DEBUG - END

        # Convert to batch
        final_data_3d_batch = torch.Tensor(data_3d_fusion).unsqueeze(0).unsqueeze(0)

        # Predict 3D
        data_3d_predicts = model_3d(final_data_3d_batch)

        # Save the results
        data_3d_output = data_3d_predicts.squeeze().squeeze().numpy()

        # TODO: Threshold
        apply_threshold(tensor=data_3d_output, threshold=0.5)
        save_name = os.path.join(output_path, os.path.basename(format_of_2d_images).replace("_<VIEW>.png", "_output"))
        convert_numpy_to_nii_gz(numpy_data=data_3d_output, save_name=save_name)


def test_single_predict():
    format_of_2d_images = os.path.join(CROPPED_PATH, "preds_fixed_2d_v6", "PA000005_vessel_02584_<VIEW>.png")
    output_path = r"./predict_results"
    single_predict(format_of_2d_images=format_of_2d_images, output_path=output_path)


def full_predict():
    input_folder = os.path.join(CROPPED_PATH, "preds_2d_v6")
    input_format = r"PA000005_vessel"
    format_paths = pathlib.Path(input_folder).rglob(f"{input_format}_*.png")
    format_of_2d_images_set = set()
    for format_path in format_paths:
        extract_format = str(format_path).rsplit("_", maxsplit=1)[0]
        format_of_2d_images_set.add(f"{extract_format}_<VIEW>.png")

    format_of_2d_images_set = sorted(list(format_of_2d_images_set))
    output_path = r"./predict_results"
    for format_of_2d_images in tqdm(format_of_2d_images_set):
        single_predict(format_of_2d_images=format_of_2d_images, output_path=output_path)


def calculate_dice_scores():
    output_folder = r"./predict_results"
    target_folder = os.path.join(CROPPED_PATH, "labels_3d_v6")

    output_format = r"PA000005_vessel_*_output"
    target_format = r"PA000005_vessel_*"

    output_format_paths = pathlib.Path(output_folder).rglob(f"{output_format}.nii.gz")
    target_format_paths = pathlib.Path(target_folder).rglob(f"{target_format}.nii.gz")
    output_paths = sorted(list(output_format_paths))
    target_paths = sorted(list(target_format_paths))

    files_count = len(output_paths)
    scores_dict = dict()
    for idx in range(files_count):
        output_path = output_paths[idx]
        target_path = target_paths[idx]

        output_nii = nib.load(output_path)
        target_nii = nib.load(target_path)

        output_data = output_nii.get_fdata()
        target_data = target_nii.get_fdata()

        dice_score = 2 * np.sum(output_data * target_data) / (np.sum(output_data) + np.sum(target_data))

        idx_format = str(output_path).rsplit("_", maxsplit=2)[1]
        scores_dict[f"PA000005_vessel_{idx_format}"] = dice_score

    pd.DataFrame(scores_dict.items()).to_csv('out.csv')


def main():
    # 1. Use model 1 on the `parse_preds_mini_cropped_v5`
    # 2. Save the results in `parse_fixed_mini_cropped_v5`
    # 3. Perform direct `logical or` on `parse_fixed_mini_cropped_v5` to get `parse_prefixed_mini_cropped_3d_v5`
    # 4. Use model 2 on the `parse_prefixed_mini_cropped_3d_v5`
    # 5. Save the results in `parse_fixed_mini_cropped_3d_v5`
    # 6. Run steps 1-5 for mini cubes and combine all the results to get the final result
    # 7. Perform cleanup on the final result (delete small connected components)

    # full_predict()
    calculate_dice_scores()


if __name__ == "__main__":
    # 1
    # TODO: validate that after the 2d projection reconstruct, applying again 2d projection give the same results (artificial tests)
    # TODO: add plots of 3d data in matplotlib
    # TODO: creation of the 3d data of fixed 3d pred


    # 2
    # TODO: add 45 degrees projections
    # TODO: use the ground truth to create create a holes


    # 3
    # TODO: create the classification models


    # completed
    # TODO: run test of different inputs (100 examples)
    # TODO: show best 10, worst 10, median 10

    # TODO: another note for next stage:
    # 1. Add label on number of connected components inside a 3D volume
    # 2. Use the label to add a task for the model to predict the number of connected components
    parser = argparse.ArgumentParser(description='Main function to call training for different AutoEncoders')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='enables CUDA predicting')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--results-path', type=str, default='./results/predict_pipeline', metavar='N',
                        help='Where to store images')
    parser.add_argument('--weights-filepath', type=str, default='./weights/Network.pth', metavar='N',
                        help='Which dataset to use')

    args = parser.parse_args()

    main()