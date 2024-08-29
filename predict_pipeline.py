import argparse
import os
import torch
import cv2
import numpy as np
import nibabel as nib
from torchvision import transforms
import matplotlib.pyplot as plt

from models.ae_v2_model import Network
from models.ae_3d_v2_model import Network3D


def convert_numpy_to_nii_gz(numpy_array, save_name=None):
    ct_nii_gz = nib.Nifti1Image(numpy_array, affine=np.eye(4))
    if save_name is not None:
        nib.save(ct_nii_gz, f"{save_name}.nii.gz")
    return ct_nii_gz


def reverse_rotations(numpy_image, view_type):
    # Convert to 3D
    data_3d = np.zeros((numpy_image.shape[0], numpy_image.shape[0], numpy_image.shape[0]), dtype=np.uint8)
    for i in range(numpy_image.shape[0]):
        for j in range(numpy_image.shape[1]):
            gray_value = int(numpy_image[i, j])
            if gray_value > 0:
                rescale_gray_value = int(numpy_image.shape[0] * (1 - (gray_value / 255)))

                if view_type in ["front", "back"]:
                    data_3d[i, j, rescale_gray_value] = 255
                elif view_type in ["top", "bottom"]:
                    data_3d[rescale_gray_value, i, j] = 255
                elif view_type in ["right", "left"]:
                    data_3d[i, rescale_gray_value, j] = 255
                else:
                    raise ValueError("Invalid view type")

    # Reverse the rotations
    if view_type == "front":
        pass

    if view_type == "back":
        data_3d = np.rot90(data_3d, k=2, axes=(2, 1))

    if view_type == "top":
        data_3d = np.rot90(data_3d, k=1, axes=(2, 1))

    if view_type == "bottom":
        data_3d = np.rot90(data_3d, k=2, axes=(1, 0))
        data_3d = np.rot90(data_3d, k=1, axes=(2, 1))

    if view_type == "right":
        data_3d = np.flip(data_3d, axis=1)

    if view_type == "left":
        data_3d = np.rot90(data_3d, k=2, axes=(2, 1))
        data_3d = np.flip(data_3d, axis=1)

    # Reverse the initial rotations
    data_3d = np.flip(data_3d, axis=1)
    data_3d = np.rot90(data_3d, k=1, axes=(2, 1))
    data_3d = np.rot90(data_3d, k=1, axes=(2, 0))

    return data_3d

def apply_threshold(tensor, threshold):
    tensor[tensor >= threshold] = 1.0
    tensor[tensor < threshold] = 0.0

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
        # Prepare 2D data
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

        # Prepare 3D data
        data_2d_predicts = data_2d_predicts.numpy()
        data_3d_list = list()
        for idx, image_view in enumerate(images_6_views):
            numpy_image = data_2d_predicts[idx].squeeze() * 255
            data_3d = reverse_rotations(numpy_image, image_view)
            data_3d_list.append(data_3d)

        # TODO: DEBUG
        data_2d = data_2d.numpy()

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

        final_data_3d = data_3d_list[0]
        for i in range(1, len(data_3d_list)):
            final_data_3d = np.logical_or(final_data_3d, data_3d_list[i])
        final_data_3d = final_data_3d.astype(np.float32)

        # TODO: DEBUG
        save_name = os.path.join(output_path, os.path.basename(format_of_2d_images).replace("_<VIEW>.png", "_input"))
        convert_numpy_to_nii_gz(numpy_array=final_data_3d, save_name=save_name)

        # Convert to batch
        final_data_3d_batch = torch.Tensor(final_data_3d).unsqueeze(0).unsqueeze(0)

        # Predict 3D
        data_3d_predicts = model_3d(final_data_3d_batch)

        # Save the results
        data_3d_output = data_3d_predicts.squeeze().squeeze().numpy()

        # TODO: Threshold
        apply_threshold(data_3d_output, 0.1)
        save_name = os.path.join(output_path, os.path.basename(format_of_2d_images).replace("_<VIEW>.png", "_output"))
        convert_numpy_to_nii_gz(numpy_array=data_3d_output, save_name=save_name)


def main():
    # 1. Use model 1 on the `parse_preds_mini_cropped_v5`
    # 2. Save the results in `parse_fixed_mini_cropped_v5`
    # 3. Perform direct `logical or` on `parse_fixed_mini_cropped_v5` to get `parse_prefixed_mini_cropped_3d_v5`
    # 4. Use model 2 on the `parse_prefixed_mini_cropped_3d_v5`
    # 5. Save the results in `parse_fixed_mini_cropped_3d_v5`
    # 6. Run steps 1-5 for mini cubes and combine all the results to get the final result

    format_of_2d_images = r"./tools/data/parse_preds_mini_cropped_v5/PA000005_vessel_02584_<VIEW>.png"
    output_path = r"./predict_results"
    single_predict(format_of_2d_images=format_of_2d_images, output_path=output_path)


if __name__ == "__main__":
    # TODO: run test of different inputs (100 examples)
    # TODO: show best 10, worst 10, median 10

    # TODO: another note for next stage:
    # 1. Add label on number of connected components inside a 3D volume
    # 2. Use the label to add a task for the model to predict the number of connected components
    parser = argparse.ArgumentParser(description='Main function to call training for different AutoEncoders')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--results-path', type=str, default='results/', metavar='N',
                        help='Where to store images')
    parser.add_argument('--dataset', type=str, default='MNIST', metavar='N',
                        help='Which dataset to use')
    parser.add_argument('--weights-filepath', type=str, default='./weights/Network.pth', metavar='N',
                        help='Which dataset to use')

    args = parser.parse_args()

    main()