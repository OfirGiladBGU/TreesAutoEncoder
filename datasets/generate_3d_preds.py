import os
import pathlib
from tqdm import tqdm
import numpy as np
import random
from scipy.ndimage import convolve

from datasets.dataset_utils import convert_data_file_to_numpy, convert_numpy_to_data_file
from datasets.dataset_list import DATA_PATH


DATASET_PATH = os.path.join(DATA_PATH, "Pipes3DGeneratorTree")


###################
# Generate Labels #
###################
def expand_connected_neighbors(array: np.ndarray) -> np.ndarray:
    """
    Expands the value of `1` to all 6-connected neighbors in a 3D numpy array.
    Args:
        array (np.ndarray): A 3D numpy array with binary values (0 and 1).

    Returns:
        np.ndarray: A 3D numpy array with the neighbors of all `1` voxels set to `1`.
    """
    # V1: Very slow method
    # Get the shape of the array
    # x_max, y_max, z_max = array.shape
    #
    # # Create a copy of the array to store results
    # expanded_array = array.copy()
    #
    # # Iterate over the array to find all 1's and expand to their 6 neighbors
    # for x in range(x_max):
    #     for y in range(y_max):
    #         for z in range(z_max):
    #             if array[x, y, z] == 1:
    #                 # Update neighbors in 6 directions
    #                 if x > 0: expanded_array[x - 1, y, z] = 1
    #                 if x < x_max - 1: expanded_array[x + 1, y, z] = 1
    #                 if y > 0: expanded_array[x, y - 1, z] = 1
    #                 if y < y_max - 1: expanded_array[x, y + 1, z] = 1
    #                 if z > 0: expanded_array[x, y, z - 1] = 1
    #                 if z < z_max - 1: expanded_array[x, y, z + 1] = 1

    # V2: Faster method
    # Define a kernel for 6-connected neighbors
    kernel = np.array([
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    ])

    # Apply the convolution
    convolved = convolve(array, kernel, mode='constant', cval=0)

    # Threshold the result to binary (0 or 1)
    expanded_array = (convolved > 0).astype(np.uint8)

    return expanded_array


# Create new 'labels' folder with numpy data
def convert_originals_data_to_labels_data(save_as_npy: bool = False, increase_density: bool = False):
    """
    Converts the original data to discrete data for numpy array, and then save the result in labels folder.
    :param save_as_npy:
    :param increase_density:
    :return:
    """
    input_folder = os.path.join(DATASET_PATH, "originals")
    output_folder = os.path.join(DATASET_PATH, "labels")

    os.makedirs(output_folder, exist_ok=True)
    data_filepaths = sorted(pathlib.Path(input_folder).rglob("*.*"))

    filepaths_count = len(data_filepaths)
    for filepath_idx in tqdm(range(filepaths_count)):
        # Get index data:
        data_filepath = data_filepaths[filepath_idx]
        numpy_data = convert_data_file_to_numpy(data_filepath=data_filepath)

        if increase_density is True:
            numpy_data = expand_connected_neighbors(array=numpy_data)

        # Save data:
        save_filename = os.path.join(output_folder, data_filepath.stem)

        if save_as_npy is True:
            data_filepath = f"{data_filepath}.npy"
        convert_numpy_to_data_file(numpy_data=numpy_data, source_data_filepath=data_filepath,
                                   save_filename=save_filename)


##################
# Generate Preds #
##################
def generate_circular_holes(numpy_data: np.ndarray):
    num_of_centers = 5
    white_points = np.argwhere(numpy_data > 0.5)
    if len(white_points) > 0:
        for _ in range(num_of_centers):
            radius = random.randint(3, 5)

            # Randomly select one of the non-zero points
            random_point = random.choice(white_points)
            x, y, z = random_point[0], random_point[1], random_point[2]  # Get the coordinates

            for i in range(max(0, x - radius), min(numpy_data.shape[0], x + radius + 1)):
                for j in range(max(0, y - radius), min(numpy_data.shape[1], y + radius + 1)):
                    for k in range(max(0, z - radius), min(numpy_data.shape[2], z + radius + 1)):
                        if (i - x) ** 2 + (j - y) ** 2 + (k - z) ** 2 <= radius ** 2:
                            numpy_data[i, j, k] = 0

    return numpy_data


def generate_plane_holes(numpy_data: np.ndarray):
    num_of_centers = 5
    white_points = np.argwhere(numpy_data > 0.5)
    if len(white_points) > 0:
        for _ in range(num_of_centers):
            size = random.randint(0, 1)

            # Randomly select one of the non-zero points
            random_point = random.choice(white_points)
            x, y, z = random_point[0], random_point[1], random_point[2]  # Get the coordinates

            # Define cube boundaries
            x_min = max(0, x - size)
            x_max = min(numpy_data.shape[0], x + size + 1)
            y_min = max(0, y - size)
            y_max = min(numpy_data.shape[1], y + size + 1)
            z_min = max(0, z - size)
            z_max = min(numpy_data.shape[2], z + size + 1)

            # Set the cube to black
            numpy_data[x_min:x_max, :, :] = 0  # Modify along the YZ planes
            numpy_data[:, y_min:y_max, :] = 0  # Modify along the XZ planes
            numpy_data[:, :, z_min:z_max] = 0  # Modify along the XY planes

            # Set all points on the same x, y, and z planes to black
            numpy_data[x, :, :] = 0  # Set all points on the plane parallel to YZ to black
            numpy_data[:, y, :] = 0  # Set all points on the plane parallel to XZ to black
            numpy_data[:, :, z] = 0  # Set all points on the plane parallel to XY to black


# Create new 'preds' folder with holes in numpy data
def convert_labels_data_to_preds_data(save_as_npy: bool = False):
    """
    Converts the labels data to preds data with holes in numpy array, and then save the result in preds folder.
    :param save_as_npy:
    :return:
    """
    input_folder = os.path.join(DATASET_PATH, "labels")
    output_folder = os.path.join(DATASET_PATH, "preds")

    os.makedirs(output_folder, exist_ok=True)
    data_filepaths = sorted(pathlib.Path(input_folder).rglob("*.*"))

    filepaths_count = len(data_filepaths)
    for filepath_idx in tqdm(range(filepaths_count)):
        # Get index data:
        data_filepath = data_filepaths[filepath_idx]
        numpy_data = convert_data_file_to_numpy(data_filepath=data_filepath)

        # Generate holes:
        # TODO: implement (Use different method)
        # generate_circular_holes(numpy_data=numpy_data)
        generate_plane_holes(numpy_data=numpy_data)

        # Save data:
        save_filename = os.path.join(output_folder, data_filepath.stem)

        if save_as_npy is True:
            data_filepath = f"{data_filepath}.npy"
        convert_numpy_to_data_file(numpy_data=numpy_data, source_data_filepath=data_filepath,
                                   save_filename=save_filename)


def main():
    # convert_originals_data_to_labels_data(save_as_npy=True, increase_density=False)
    convert_labels_data_to_preds_data(save_as_npy=True)


if __name__ == '__main__':
    main()
