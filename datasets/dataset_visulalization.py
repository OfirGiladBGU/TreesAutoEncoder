import numpy as np
import matplotlib.pyplot as plt

from datasets.dataset_utils import convert_nii_gz_to_numpy
from datasets.dataset_list import CROPPED_PATH


def plot_3d_data(data_3d, idx):
    fig_name = f"3d_plot_{idx}"
    print(f"Data shape: {data_3d.shape}")

    # Downsample the images
    downsample_factor = 1
    data_downsampled = data_3d[::downsample_factor, ::downsample_factor, ::downsample_factor]

    # Get the indices of non-zero values in the downsampled array
    nonzero_indices = np.where(data_downsampled != 0)

    # Create a figure and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the cubes based on the non-zero indices
    ax.bar3d(nonzero_indices[2], nonzero_indices[0], nonzero_indices[1], 1, 1, 1, color='b')
    # ax.bar3d(nonzero_indices[0], nonzero_indices[1], nonzero_indices[2], 1, 1, 1, color='b')

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Display the plot
    plt.title('3d plot')
    plt.savefig(fig_name)
    plt.close('all')


def test_plot_3d_data():
    data_file = r"C:\Users\ofirg\PycharmProjects\TreesAutoEncoder\data\cropped_data\labels_3d_reconstruct_v6\PA000005_04510.nii.gz"
    # data_file = r"C:\Users\ofirg\PycharmProjects\TreesAutoEncoder\data\cropped_data\preds_3d_reconstruct_v6\PA000005_04510.nii.gz"

    # data_file = r"C:\Users\ofirg\PycharmProjects\TreesAutoEncoder\data\cropped_data\labels_3d_v6\PA000005_04510.nii.gz"
    # data_file = r"C:\Users\ofirg\PycharmProjects\TreesAutoEncoder\data\cropped_data\preds_3d_v6\PA000005_04510.nii.gz"

    # data_file = r"C:\Users\ofirg\PycharmProjects\TreesAutoEncoder\data\cropped_data\preds_3d_fusion_v6\PA000005_04510.nii.gz"
    # data_file = r"C:\Users\ofirg\PycharmProjects\TreesAutoEncoder\data\cropped_data\preds_fixed_3d_fusion_v6\PA000005_04510.nii.gz"

    output_idx = 1
    numpy_data = convert_nii_gz_to_numpy(data_filepath=data_file)
    plot_3d_data(data_3d=numpy_data, idx=output_idx)


def main():
    test_plot_3d_data()


if __name__ == "__main__":
    # TODO:
    # 1. Make sure "skel_np" folder is present in the root directory
    main()
