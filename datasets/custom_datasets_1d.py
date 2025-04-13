import argparse
import os
import pathlib
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import pandas as pd

from datasets_forge.dataset_configurations import (TRAIN_LOG_PATH, V1_1D_DATASETS, IMAGES_6_VIEWS)
from datasets.dataset_utils import get_data_file_stem, convert_data_file_to_numpy, validate_data_paths


# 1 2D input + 1 1D target
class TreesCustomDatasetV1(Dataset):
    def __init__(self, args: argparse.Namespace, data_paths: list, transform=None):
        self.args = args
        self.data_paths = data_paths
        self.transform = transform
        self.to_tensor = transforms.ToTensor()

        if not os.path.exists(TRAIN_LOG_PATH):
            raise FileNotFoundError(f"File not found: {TRAIN_LOG_PATH}")
        else:
            self.log_data = pd.read_csv(TRAIN_LOG_PATH)  # filter invalid data

        self.paths_count = len(data_paths)
        if not (self.paths_count == 1):
            raise ValueError("Invalid number of data paths")

        # current_count > 0:
        self.data_files1 = pathlib.Path(data_paths[0]).rglob("*.png")
        self.data_files1 = sorted(self.data_files1)
        self.data_classes = []

        ###############
        # Get classes #
        ###############

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

        for filepath_idx in range(filepaths_count):
            data_file1 = str(self.data_files1[filepath_idx])
            data_file1_filename = get_data_file_stem(data_filepath=data_file1)
            file1_conditions = [
                # data_file1_filename not in non_valid_filenames,  # TODO (Optional)
                data_file1_filename not in non_advance_valid_filenames
            ]
            if all(file1_conditions):
                self.data_classes.append(1.0)
            else:
                self.data_classes.append(0.0)

        self.dataset_count = len(self.data_files1)

    def __len__(self):
        return self.dataset_count

    def __getitem__(self, idx):
        item = tuple()

        data_file1 = str(self.data_files1[idx])
        numpy_2d_data1 = convert_data_file_to_numpy(data_filepath=data_file1)

        numpy_2d_data1 = self.to_tensor(numpy_2d_data1)
        if self.transform is not None:
            numpy_2d_data1 = self.transform(numpy_2d_data1)

        if self.paths_count == 1:
            data_class = torch.Tensor([self.data_classes[idx]])
            item += (numpy_2d_data1, data_class)
        else:
            pass

        return item


class TreesCustomDataset1D:
    def __init__(self, args: argparse.Namespace, data_paths, transform=None):
        validate_data_paths(data_paths=data_paths)
        self.args = args
        self.data_paths = data_paths
        self.transform = transform
        self._init_subsets()

    def _init_subsets(self):
        if self.args.dataset in V1_1D_DATASETS:
            trees_dataset = TreesCustomDatasetV1(
                args=self.args,
                data_paths=self.data_paths,
                transform=self.transform
            )
        else:
            raise Exception("Dataset not available in 'Custom Dataset 1D'")

        # TODO: Perform split based on the 3D files count

        dataset_size = len(trees_dataset)
        train_size = int(dataset_size * 0.9)
        val_size = dataset_size - train_size

        # self.train_subset, self.test_subset = torch.utils.data.random_split(trees_dataset, [train_size, val_size])

        # Non random split
        self.train_subset = Subset(trees_dataset, indices=range(0, train_size))
        self.test_subset = Subset(trees_dataset, indices=range(train_size, train_size + val_size))
