import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import itertools

from datasets.dataset_utils import convert_nii_gz_to_numpy
from datasets.dataset_list import CROPPED_PATH, RESULTS_PATH


def matplotlib_plot_3d(data_3d, save_filename):
    print(f"Filename: {save_filename}, Data shape: {data_3d.shape}")

    # Downsample the images
    downsample_factor = 1
    data_downsampled = data_3d[::downsample_factor, ::downsample_factor, ::downsample_factor]

    # Get the indices of non-zero values in the downsampled array
    nonzero_indices = np.where(data_downsampled != 0)

    # Plot the cubes based on the non-zero indices
    for permutation in list(itertools.permutations(iterable=[0, 1, 2], r=3)):
        # Create a figure and a 3D axis
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Get the permutation
        i, j, k = permutation
        ax.bar3d(nonzero_indices[i], nonzero_indices[j], nonzero_indices[k], 1, 1, 1, color='b')

        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Display the plot
        permutation_save_filename = f"{save_filename}_({i},{j},{k})"
        plt.title('3d plot')
        plt.savefig(permutation_save_filename)
        plt.close('all')


def plot_3d_data():
    data_3d_name = "PA000005_04510"

    folder_paths = {
        "labels_3d": os.path.join(CROPPED_PATH, "labels_3d_v6"),
        "labels_3d_reconstruct": os.path.join(CROPPED_PATH, "labels_3d_reconstruct_v6"),

        "preds_3d": os.path.join(CROPPED_PATH, "preds_3d_v6"),
        "preds_3d_reconstruct": os.path.join(CROPPED_PATH, "preds_3d_reconstruct_v6"),
        "preds_3d_fusion": os.path.join(CROPPED_PATH, "preds_3d_fusion_v6"),

        "preds_fixed_3d": os.path.join(CROPPED_PATH, "preds_fixed_3d_v6"),
        "preds_fixed_3d_reconstruct": os.path.join(CROPPED_PATH, "preds_fixed_3d_reconstruct_v6"),
        "preds_fixed_3d_fusion": os.path.join(CROPPED_PATH, "preds_fixed_3d_fusion_v6")
    }

    for key, value in folder_paths.items():
        save_path = os.path.join(RESULTS_PATH, "visualisations", key)
        os.makedirs(name=save_path, exist_ok=True)
        save_filename = os.path.join(str(save_path), f"{data_3d_name}")

        data_3d_filepath = list(pathlib.Path(value).rglob(f"{data_3d_name}*.nii.gz"))[0]
        numpy_3d_data = convert_nii_gz_to_numpy(data_filepath=data_3d_filepath)

        matplotlib_plot_3d(data_3d=numpy_3d_data, save_filename=save_filename)


def main():
    plot_3d_data()


if __name__ == "__main__":
    main()
