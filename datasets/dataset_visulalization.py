import numpy as np
import matplotlib.pyplot as plt
import os
import itertools
import cv2

from datasets.dataset_utils import convert_nii_gz_to_numpy
from datasets.dataset_list import CROPPED_PATH, VISUALIZATION_RESULTS_PATH, RESULTS_PATH, PREDICT_PIPELINE_RESULTS_PATH


####################
# 3D visualization #
####################
def matplotlib_plot_3d(data_3d: np.ndarray, save_filename):
    print(f"Save filename: '{save_filename}*'\nData shape: '{data_3d.shape}'\n")
    data_3d = data_3d
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


def single_plot_3d():
    # data_3d_filepath = os.path.join(RESULTS_PATH, "predict_pipeline", "output_3d", "PA000005_05352_input.nii.gz")
    # data_3d_filepath = os.path.join(CROPPED_PATH, "labels_3d_v6", "PA000005_05352.nii.gz")
    data_3d_filepath = os.path.join(CROPPED_PATH, "preds_3d_v6", "PA000005_05352.nii.gz")
    numpy_3d_data = convert_nii_gz_to_numpy(data_filepath=data_3d_filepath)

    save_path = os.path.join(RESULTS_PATH, "single_predict")
    os.makedirs(name=save_path, exist_ok=True)
    save_name = os.path.join(save_path, "pred_og")

    matplotlib_plot_3d(data_3d=numpy_3d_data, save_filename=save_name)



def full_plot_3d(include_pipeline_results=False):
    data_3d_basename = "PA000005_11899"

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
        save_path = os.path.join(VISUALIZATION_RESULTS_PATH, key)
        os.makedirs(name=save_path, exist_ok=True)
        save_filename = os.path.join(str(save_path), f"{data_3d_basename}")

        data_3d_filepath = os.path.join(value, f"{data_3d_basename}.nii.gz")
        numpy_3d_data = convert_nii_gz_to_numpy(data_filepath=data_3d_filepath)

        matplotlib_plot_3d(data_3d=numpy_3d_data, save_filename=save_filename)

    if include_pipeline_results is True:
        folder_path = os.path.join(PREDICT_PIPELINE_RESULTS_PATH, "output_3d")

        result_types = ["input", "output"]

        for result_type in result_types:
            save_path = os.path.join(VISUALIZATION_RESULTS_PATH, f"pipeline_{result_type}")
            os.makedirs(name=save_path, exist_ok=True)
            save_filename = os.path.join(str(save_path), f"{data_3d_basename}")

            data_3d_filepath = os.path.join(folder_path, f"{data_3d_basename}_{result_type}.nii.gz")
            numpy_3d_data = convert_nii_gz_to_numpy(data_filepath=data_3d_filepath)

            matplotlib_plot_3d(data_3d=numpy_3d_data, save_filename=save_filename)


####################
# 2D visualization #
####################
IMAGES_6_VIEWS = ['top', 'bottom', 'front', 'back', 'left', 'right']


def matplotlib_plot_2d(save_filepath, data_2d_list):
    columns = 6
    rows = 1
    fig = plt.figure(figsize=(columns + 0.5, rows + 0.5))
    ax = list()

    # 2D Input
    for j in range(columns):
        ax.append(fig.add_subplot(rows, columns, 0 * columns + j + 1))
        numpy_image = data_2d_list[j]
        numpy_image = numpy_image
        plt.imshow(numpy_image, cmap='gray')
        ax[j].set_title(f"View {IMAGES_6_VIEWS[j]}:")

    fig.tight_layout()
    plt.savefig(save_filepath)
    plt.close(fig)


def full_plot_2d():
    data_3d_basename = "PA000005_11899"

    folder_paths = {
        "labels_2d": os.path.join(CROPPED_PATH, "labels_2d_v6"),
        "preds_2d": os.path.join(CROPPED_PATH, "preds_2d_v6"),
        "preds_fixed_2d": os.path.join(CROPPED_PATH, "preds_fixed_2d_v6")
    }

    for key, value in folder_paths.items():
        format_of_2d_images = os.path.join(value, f"{data_3d_basename}_<VIEW>.png")
        # Projections 2D
        data_2d_list = list()
        for image_view in IMAGES_6_VIEWS:
            image_path = format_of_2d_images.replace("<VIEW>", image_view)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            data_2d_list.append(image)

        save_path = os.path.join(VISUALIZATION_RESULTS_PATH, key)
        os.makedirs(name=save_path, exist_ok=True)
        save_filepath = os.path.join(str(save_path), f"{data_3d_basename}")
        matplotlib_plot_2d(save_filepath=save_filepath, data_2d_list=data_2d_list)


def main():
    # single_plot_3d()
    # full_plot_3d()
    # full_plot_3d(include_pipeline_results=True)
    full_plot_2d()


if __name__ == "__main__":
    main()
