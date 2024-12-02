import os
import pathlib
from tqdm import tqdm

from datasets.dataset_utils import convert_data_file_to_numpy, convert_numpy_to_data_file
from datasets.dataset_list import DATA_PATH

DATASET_PATH = os.path.join(DATA_PATH, "PyPipes")

def generate_holes():
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

        # Save data:
        save_filename = os.path.join(output_folder, data_filepath.name)
        convert_numpy_to_data_file(numpy_data=numpy_data, source_data_filepath=data_filepath,
                                   save_filename=save_filename)


def main():
    generate_holes()


if __name__ == '__main__':
    main()
