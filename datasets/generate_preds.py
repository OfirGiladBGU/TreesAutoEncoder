import os
import pathlib
from tqdm import tqdm
import numpy as np
import random

from datasets.dataset_utils import convert_data_file_to_numpy, convert_numpy_to_data_file
from datasets.dataset_list import DATA_PATH


DATASET_PATH = os.path.join(DATA_PATH, "Pipes3DGenerator")


# Create new 'labels' folder with numpy data
def convert_original_data_to_npy():
    input_folder = os.path.join(DATASET_PATH, "originals")
    output_folder = os.path.join(DATASET_PATH, "labels")

    os.makedirs(output_folder, exist_ok=True)
    data_filepaths = sorted(pathlib.Path(input_folder).rglob("*.*"))

    filepaths_count = len(data_filepaths)
    for filepath_idx in tqdm(range(filepaths_count)):
        # Get index data:
        data_filepath = data_filepaths[filepath_idx]
        numpy_data = convert_data_file_to_numpy(data_filepath=data_filepath)

        # Save data:
        save_filename = os.path.join(output_folder, data_filepath.stem)
        convert_numpy_to_data_file(numpy_data=numpy_data, source_data_filepath="dummy.npy",
                                   save_filename=save_filename)


# Create new 'preds' folder with holes in numpy data
def generate_holes_in_data():
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
        # TODO: implement
        num_of_holes = 5
        white_points = np.argwhere(numpy_data > 0.5)
        if len(white_points) > 0:
            for _ in range(num_of_holes):
                # Randomly select one of the non-zero points
                random_point = random.choice(white_points)
                radius = random.randint(3, 5)

                x, y, z = random_point[0], random_point[1], random_point[2]  # Get the coordinates
                for i in range(max(0, x - radius), min(numpy_data.shape[0], x + radius + 1)):
                    for j in range(max(0, y - radius), min(numpy_data.shape[1], y + radius + 1)):
                        for k in range(max(0, z - radius), min(numpy_data.shape[2], z + radius + 1)):
                            if (i - x) ** 2 + (j - y) ** 2 + (k - z) ** 2 <= radius ** 2:
                                numpy_data[i, j, k] = 0

        # Save data:
        save_filename = os.path.join(output_folder, data_filepath.stem)
        convert_numpy_to_data_file(numpy_data=numpy_data, source_data_filepath="dummy.npy",
                                   save_filename=save_filename)


def main():
    # convert_original_data_to_npy()
    generate_holes_in_data()


if __name__ == '__main__':
    main()
