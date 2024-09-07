import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import cv2
import pathlib
import pandas as pd


class TreesCustomDatasetV1(Dataset):
    def __init__(self, data_paths: list, log_path=None, transform=None):
        self.data_paths = data_paths
        self.log_path = log_path
        self.transform = transform
        self.to_tensor = transforms.ToTensor()

        self.paths_count = len(data_paths)
        if not (1 <= self.paths_count <= 2):
            raise ValueError("Invalid number of data paths")
        current_count = self.paths_count

        # current_count > 0:
        self.data_files1 = pathlib.Path(data_paths[0]).rglob("*.png")
        self.data_files1 = sorted(self.data_files1)
        current_count -= 1

        if current_count > 0:
            self.data_files2 = pathlib.Path(data_paths[1]).rglob("*.png")
            self.data_files2 = sorted(self.data_files2)
            current_count -= 1

        if self.log_path is not None:
            self.log_data = pd.read_csv(self.log_path)

        self.dataset_count = len(self.data_files1)

    def __len__(self):
        return self.dataset_count

    def __getitem__(self, idx):
        item = tuple()

        numpy_2d_data2 = None
        current_count = self.paths_count

        # current_count > 0:
        data_file1 = str(self.data_files1[idx])
        numpy_2d_data1 = cv2.imread(data_file1)
        numpy_2d_data1 = cv2.cvtColor(numpy_2d_data1, cv2.COLOR_BGR2GRAY)

        numpy_2d_data1 = self.to_tensor(numpy_2d_data1)
        if self.transform is not None:
            numpy_2d_data1 = self.transform(numpy_2d_data1)
        current_count -= 1

        if current_count > 0:
            data_file2 = str(self.data_files2[idx])
            numpy_2d_data2 = cv2.imread(data_file2)
            numpy_2d_data2 = cv2.cvtColor(numpy_2d_data2, cv2.COLOR_BGR2GRAY)

            numpy_2d_data2 = self.to_tensor(numpy_2d_data2)
            if self.transform is not None:
                numpy_2d_data2 = self.transform(numpy_2d_data2)
            current_count -= 1

        if self.paths_count == 1:
            item += (numpy_2d_data1, -1)
        elif self.paths_count == 2:
            item += (numpy_2d_data1, numpy_2d_data2)

        if self.log_path is not None:
            label_local_components = self.log_data["label_local_components"][idx]
            item += (label_local_components,)

        return item


# TODO: Check if will be useful
class TreesCustomDatasetV2(Dataset):
    def __init__(self, data_paths: list, log_path=None, transform=None):
        self.data_paths = data_paths
        self.log_path = log_path
        self.transform = transform
        self.to_tensor = transforms.ToTensor()

        self.paths_count = len(data_paths)
        if not (1 <= self.paths_count <= 2):
            raise ValueError("Invalid number of data paths")

        current_count = self.paths_count

        # current_count > 0:
        self.data_files1 = pathlib.Path(data_paths[0]).rglob("*.png")
        self.data_files1 = sorted(self.data_files1)
        current_count -= 1

        if current_count > 0:
            self.data_files2 = pathlib.Path(data_paths[1]).rglob("*.png")
            self.data_files2 = sorted(self.data_files2)
            current_count -= 1

        if self.log_path is not None:
            self.log_data = pd.read_csv(self.log_path)

        self.dataset_count = int(len(self.data_files1) / 6)

    def __len__(self):
        return self.dataset_count

    def __getitem__(self, idx):
        item = tuple()

        batch1 = list()
        batch2 = list()

        data_idx = idx * 6
        for i in range(6):
            current_count = self.paths_count

            # current_count > 0:
            data_file1 = str(self.data_files1[data_idx + i])
            numpy_2d_data1 = cv2.imread(data_file1)
            numpy_2d_data1 = cv2.cvtColor(numpy_2d_data1, cv2.COLOR_BGR2GRAY)

            numpy_2d_data1 = self.to_tensor(numpy_2d_data1)
            if self.transform is not None:
                numpy_2d_data1 = self.transform(numpy_2d_data1)
            batch1.append(numpy_2d_data1)
            current_count -= 1

            if current_count > 0:
                data_file2 = str(self.data_files2[data_idx + i])
                numpy_2d_data2 = cv2.imread(data_file2)
                numpy_2d_data2 = cv2.cvtColor(numpy_2d_data2, cv2.COLOR_BGR2GRAY)

                numpy_2d_data2 = self.to_tensor(numpy_2d_data2)
                if self.transform is not None:
                    numpy_2d_data2 = self.transform(numpy_2d_data2)
                batch2.append(numpy_2d_data2)
                current_count -= 1

        if self.paths_count == 1:
            item += (batch1, -1)
        elif self.paths_count == 2:
            item += (batch1, batch2)

        if self.log_path is not None:
            label_local_components = self.log_data["label_local_components"][idx]
            item += (label_local_components,)

        return item


class TreesCustomDataloader2D:
    def __init__(self, args: argparse.Namespace, data_paths, log_path=None, transform=None):
        self.args = args
        self.data_paths = data_paths
        self.log_path = log_path
        self.transform = transform
        self._init_dataloader()

    def _init_dataloader(self):
        """
        Return Train and Val Dataloaders for the given parameters.

        Returns:
            train_dataloader: Train loader with 0.9 of the data.
            val_dataloader: Val loader with 0.1 of the data.
        """
        if self.args.dataset in ['Trees2DV1', 'Trees2DV1S']:
            tree_dataset = TreesCustomDatasetV1(
                data_paths=self.data_paths,
                log_path=self.log_path,
                transform=self.transform
            )
        elif self.args.dataset == 'Trees2DV2':
            tree_dataset = TreesCustomDatasetV2(
                data_paths=self.data_paths,
                log_path=self.log_path,
                transform=self.transform
            )
        else:
            raise Exception("Dataset not supported")

        dataset_size = len(tree_dataset)
        train_size = int(dataset_size * 0.9)
        val_size = dataset_size - train_size

        # train_data, test_data = torch.utils.data.random_split(tree_dataset, [train_size, val_size])

        # Non random split
        train_data = Subset(tree_dataset, indices=range(0, train_size))
        test_data = Subset(tree_dataset, indices=range(train_size, train_size + val_size))

        # Create dataloaders
        kwargs = dict()
        batch_size = 1
        if self.args is not None:
            kwargs = {'num_workers': 1, 'pin_memory': True} if self.args.cuda else dict()
            batch_size = self.args.batch_size

        self.train_dataloader = DataLoader(
            dataset=train_data,
            batch_size=batch_size,
            shuffle=True,
            **kwargs
        )
        self.test_dataloader = DataLoader(
            dataset=test_data,
            batch_size=batch_size,
            shuffle=False,
            **kwargs
        )

    def get_dataloader(self):
        return self.train_dataloader, self.test_dataloader
