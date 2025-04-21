import argparse
import os
import pathlib
import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms
import pandas as pd

from datasets_forge.dataset_configurations import (TRAIN_LOG_PATH, V1_2D_DATASETS, V2_2D_DATASETS, IMAGES_6_VIEWS,
                                                   APPLY_LOG_FILTER)
from datasets.dataset_utils import get_data_file_stem, convert_data_file_to_numpy, validate_data_paths


# 1 2D input + 1 2D target
class TreesCustomDatasetV1(Dataset):
    def __init__(self, args: argparse.Namespace, data_paths: list, transform=None):
        self.args = args
        self.data_paths = data_paths
        self.transform = transform
        self.to_tensor = transforms.ToTensor()

        if not os.path.exists(TRAIN_LOG_PATH):
            raise FileNotFoundError(f"File not found: {TRAIN_LOG_PATH}")
        # elif self.args.include_regression is True or APPLY_LOG_FILTER is True:
        self.log_data = pd.read_csv(filepath_or_buffer=TRAIN_LOG_PATH)

        self.paths_count = len(data_paths)
        if not (1 <= self.paths_count <= 2):
            raise ValueError("Invalid number of data paths")
        current_count = self.paths_count

        # current_count > 0:
        # self.data_files1 = pathlib.Path(data_paths[0]).rglob("*.png")
        # self.data_files1 = sorted(self.data_files1)

        self.data_files1 = []
        current_count -= 1

        if current_count > 0:
            # self.data_files2 = pathlib.Path(data_paths[1]).rglob("*.png")
            # self.data_files2 = sorted(self.data_files2)

            self.data_files2 = []
            current_count -= 1
        else:
            self.data_files2 = None

        col_0 = self.log_data.columns[0]
        for row_idx, row in self.log_data.iterrows():
            for image_view in IMAGES_6_VIEWS:
                data_file1 = os.path.join(data_paths[0], f"{row[col_0]}_{image_view}.png")
                self.data_files1.append(data_file1)

                if self.data_files2 is not None:
                    data_file2 = os.path.join(data_paths[1], f"{row[col_0]}_{image_view}.png")
                    self.data_files2.append(data_file2)

        if APPLY_LOG_FILTER is True:
            # Find invalid data
            non_valid_filenames = []
            non_advance_valid_filenames = []
            for _, row in self.log_data.iterrows():
                data_basename = row.iloc[0]
                for image_view in IMAGES_6_VIEWS:
                    if row[f"{image_view}_valid"] is False:
                        non_valid_filenames.append(f"{data_basename}_{image_view}")
                    if row[f"{image_view}_advance_valid"] is False:
                        non_advance_valid_filenames.append(f"{data_basename}_{image_view}")

            # Filter invalid data paths
            filepaths_count = len(self.data_files1)
            filtered_data_files1 = []
            filtered_data_files2 = []
            for filepath_idx in range(filepaths_count):
                data_file1 = str(self.data_files1[filepath_idx])
                data_file1_filename = get_data_file_stem(data_filepath=data_file1, relative_to=data_paths[0])
                file1_conditions = [
                    data_file1_filename not in non_valid_filenames,
                    # data_file1_filename not in non_advance_valid_filenames  # TODO (Optional)
                ]
                if all(file1_conditions):
                    filtered_data_files1.append(data_file1)

                if self.data_files2 is not None:
                    data_file2 = str(self.data_files2[filepath_idx])
                    data_file2_filename = get_data_file_stem(data_filepath=data_file2, relative_to=data_paths[1])
                    file2_conditions = [
                        data_file2_filename not in non_valid_filenames,
                        # data_file2_filename not in non_advance_valid_filenames  # TODO (Optional)
                    ]
                    if all(file2_conditions):
                        filtered_data_files2.append(data_file2)

            # Update data files
            self.data_files1 = filtered_data_files1
            if self.data_files2 is not None:
                self.data_files2 = filtered_data_files2

        self.dataset_count = len(self.data_files1)

    def __len__(self):
        return self.dataset_count

    def __getitem__(self, idx):
        item = tuple()

        numpy_2d_data2 = None

        data_file1 = str(self.data_files1[idx])
        numpy_2d_data1 = convert_data_file_to_numpy(data_filepath=data_file1)

        numpy_2d_data1 = self.to_tensor(numpy_2d_data1)
        if self.transform is not None:
            numpy_2d_data1 = self.transform(numpy_2d_data1)

        if self.data_files2 is not None:
            data_file2 = str(self.data_files2[idx])
            numpy_2d_data2 = convert_data_file_to_numpy(data_filepath=data_file2)

            numpy_2d_data2 = self.to_tensor(numpy_2d_data2)
            if self.transform is not None:
                numpy_2d_data2 = self.transform(numpy_2d_data2)

        if self.paths_count == 1:
            item += (numpy_2d_data1, -1)
        elif self.paths_count == 2:
            item += (numpy_2d_data1, numpy_2d_data2)
        else:
            pass

        if self.args.include_regression is True:
            label_local_components = self.log_data["label_local_components"][idx]
            item += (label_local_components,)

        return item


# 6 2D inputs + 6 2D targets
class TreesCustomDatasetV2(Dataset):
    def __init__(self, args: argparse.Namespace, data_paths: list, transform=None):
        self.args = args
        self.data_paths = data_paths
        self.transform = transform
        self.to_tensor = transforms.ToTensor()

        if not os.path.exists(TRAIN_LOG_PATH):
            raise FileNotFoundError(f"File not found: {TRAIN_LOG_PATH}")
        # elif self.args.include_regression is True:
        self.log_data = pd.read_csv(TRAIN_LOG_PATH)

        self.paths_count = len(data_paths)
        if not (1 <= self.paths_count <= 2):
            raise ValueError("Invalid number of data paths")
        current_count = self.paths_count

        # current_count > 0:
        # self.data_files1 = pathlib.Path(data_paths[0]).rglob("*.png")
        # self.data_files1 = sorted(self.data_files1)

        self.data_files1 = []
        current_count -= 1

        if current_count > 0:
            # self.data_files2 = pathlib.Path(data_paths[1]).rglob("*.png")
            # self.data_files2 = sorted(self.data_files2)

            self.data_files2 = []
            current_count -= 1
        else:
            self.data_files2 = None

        col_0 = self.log_data.columns[0]
        for row_idx, row in self.log_data.iterrows():
            for image_view in IMAGES_6_VIEWS:
                data_file1 = os.path.join(data_paths[0], f"{row[col_0]}_{image_view}.png")
                self.data_files1.append(data_file1)

                if self.data_files2 is not None:
                    data_file2 = os.path.join(data_paths[1], f"{row[col_0]}_{image_view}.png")
                    self.data_files2.append(data_file2)

        self.dataset_count = int(len(self.data_files1) / 6)

    def __len__(self):
        return self.dataset_count

    def __getitem__(self, idx):
        item = tuple()

        batch1 = list()
        batch2 = list()

        data_idx = idx * 6
        for i in range(6):
            data_file1 = str(self.data_files1[data_idx + i])
            numpy_2d_data1 = convert_data_file_to_numpy(data_filepath=data_file1)

            numpy_2d_data1 = self.to_tensor(numpy_2d_data1)
            if self.transform is not None:
                numpy_2d_data1 = self.transform(numpy_2d_data1)
            numpy_2d_data1 = numpy_2d_data1.squeeze(0)
            batch1.append(numpy_2d_data1)

            if self.data_files2 is not None:
                data_file2 = str(self.data_files2[data_idx + i])
                numpy_2d_data2 = convert_data_file_to_numpy(data_filepath=data_file2)

                numpy_2d_data2 = self.to_tensor(numpy_2d_data2)
                if self.transform is not None:
                    numpy_2d_data2 = self.transform(numpy_2d_data2)
                numpy_2d_data2 = numpy_2d_data2.squeeze(0)
                batch2.append(numpy_2d_data2)

        batch1 = torch.stack(batch1)
        if self.paths_count == 1:
            item += (batch1, -1)
        elif self.paths_count == 2:
            batch2 = torch.stack(batch2)
            item += (batch1, batch2)
        else:
            pass

        if self.args.include_regression is True:
            label_local_components = self.log_data["label_local_components"][idx]
            item += (label_local_components,)

        return item


class TreesCustomDataset2D:
    def __init__(self, args: argparse.Namespace, data_paths, transform=None):
        validate_data_paths(data_paths=data_paths)
        self.args = args
        self.data_paths = data_paths
        self.transform = transform
        self._init_subsets()

    def _init_subsets(self):
        if self.args.dataset in V1_2D_DATASETS:
            trees_dataset = TreesCustomDatasetV1(
                args=self.args,
                data_paths=self.data_paths,
                transform=self.transform
            )
        elif self.args.dataset in V2_2D_DATASETS:
            trees_dataset = TreesCustomDatasetV2(
                args=self.args,
                data_paths=self.data_paths,
                transform=self.transform
            )
        else:
            raise Exception("Dataset not available in 'Custom Dataset 2D'")

        # TODO: Perform split based on the 3D files count

        # # Option 1: Split by files percentage
        # dataset_size = len(trees_dataset)
        # train_size = int(dataset_size * 0.9)
        # test_size = dataset_size - train_size
        # train_indices = range(0, train_size)
        # test_indices = range(train_size, train_size + test_size)

        # # self.train_subset, self.test_subset = torch.utils.data.random_split(trees_dataset, [train_size, test_size])

        # Option 2: Split based on 3D files
        index_3d_uniques = trees_dataset.log_data["index_3d"].unique()
        index_3d_split_index = round(len(index_3d_uniques) * 0.9)
        data_3d_stem_list = [data_3d_stem[1:] for data_3d_stem in index_3d_uniques]

        train_stems = data_3d_stem_list[:index_3d_split_index]
        # test_stems = data_3d_stem_list[index_3d_split_index:]

        train_indices = []
        test_indices = []
        for idx in range(len(trees_dataset)):
            if isinstance(trees_dataset, TreesCustomDatasetV2):
                file_idx = idx * len(IMAGES_6_VIEWS)
            else:
                file_idx = idx

            data_files2_stem = get_data_file_stem(
                data_filepath=trees_dataset.data_files2[file_idx],
                relative_to=trees_dataset.data_paths[1]
            )

            # Format: {output_idx}_{cube_idx}_{image_view}{ext}
            data_files2_output_idx = data_files2_stem.rsplit(sep='_', maxsplit=2)[0]
            if data_files2_output_idx in train_stems:
                train_indices.append(idx)
            else:
                test_indices.append(idx)

        # Non random split
        self.train_subset = Subset(dataset=trees_dataset, indices=train_indices)
        self.test_subset = Subset(dataset=trees_dataset, indices=test_indices)
