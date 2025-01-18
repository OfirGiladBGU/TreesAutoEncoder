import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import pathlib
import pandas as pd

from datasets.dataset_utils import convert_data_file_to_numpy, validate_data_paths


V1_3D_DATASETS = ['Trees3DV1']
V2_3D_DATASETS = ['Trees3DV2', 'Trees3DV2M', 'Trees3DV3', 'Trees3DV4']


# 6 2D inputs + 1 3D target
class TreesCustomDataset3DV1(Dataset):
    def __init__(self, data_paths: list, log_path=None, transform2d=None, transform3d=None):
        self.data_paths = data_paths
        self.log_path = log_path
        self.transform2d = transform2d
        self.transform3d = transform3d
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
            self.data_files2 = pathlib.Path(data_paths[1]).rglob("*.*")
            self.data_files2 = sorted(self.data_files2)
            current_count -= 1

        if self.log_path is not None:
            self.log_data = pd.read_csv(self.log_path)

        self.scans_count = len(self.data_files2)

    def __len__(self):
        return self.scans_count

    def __getitem__(self, idx):
        item = tuple()

        current_count = self.paths_count

        # current_count > 0:
        batch = list()
        target = None

        data_idx = idx * 6
        for i in range(6):
            data_file1 = str(self.data_files1[data_idx + i])
            numpy_2d_data1 = convert_data_file_to_numpy(data_filepath=data_file1)

            numpy_2d_data1 = self.to_tensor(numpy_2d_data1)
            if self.transform2d is not None:
                numpy_2d_data1 = self.transform2d(numpy_2d_data1)
            batch.append(numpy_2d_data1)
        batch = torch.stack(batch)
        current_count -= 1

        # 6 images + 1 3D data
        if current_count > 0:
            data_file2 =  str(self.data_files2[idx])
            numpy_3d_data2 = convert_data_file_to_numpy(data_filepath=data_file2)
            numpy_3d_data2 = numpy_3d_data2.astype(np.float32)

            numpy_3d_data2 = torch.Tensor(numpy_3d_data2)
            if self.transform3d is not None:
                numpy_3d_data2 = self.transform3d(numpy_3d_data2)
            target = torch.stack([numpy_3d_data2])

        if self.paths_count == 1:
            item += (batch, -1)
        elif self.paths_count == 2:
            item += (batch, target)

        if self.log_path is not None:
            label_local_components = self.log_data["label_local_components"][idx]
            item += (label_local_components,)

        return item


# 1 3D input + 1 3D target
class TreesCustomDataset3DV2(Dataset):
    def __init__(self, data_paths: list, log_path=None, transform3d=None):
        self.data_paths = data_paths
        self.log_path = log_path
        self.transform3d = transform3d

        self.paths_count = len(data_paths)
        if not (1 <= self.paths_count <= 2):
            raise ValueError("Invalid number of data paths")
        current_count = self.paths_count

        # current_count > 0:
        self.data_files1 = pathlib.Path(data_paths[0]).rglob("*.*")
        self.data_files1 = sorted(self.data_files1)
        current_count -= 1

        if current_count > 0:
            self.data_files2 = pathlib.Path(data_paths[1]).rglob("*.*")
            self.data_files2 = sorted(self.data_files2)
            current_count -= 1

        if self.log_path is not None:
            self.log_data = pd.read_csv(self.log_path)

        self.scans_count = len(self.data_files2)

    def __len__(self):
        return self.scans_count

    def __getitem__(self, idx):
        item = tuple()

        current_count = self.paths_count

        # current_count > 0:
        numpy_3d_data2 = None

        data_file1 = str(self.data_files1[idx])
        numpy_3d_data1 = convert_data_file_to_numpy(data_filepath=data_file1)
        numpy_3d_data1 = numpy_3d_data1.astype(np.float32)

        numpy_3d_data1 = torch.Tensor(numpy_3d_data1)
        if self.transform3d is not None:
            numpy_3d_data1 = self.transform3d(numpy_3d_data1)
        numpy_3d_data1 = numpy_3d_data1.unsqueeze(0)
        current_count -= 1

        # 1 3D input + 1 3D target
        if current_count > 0:
            data_file2 = str(self.data_files2[idx])
            numpy_3d_data2 = convert_data_file_to_numpy(data_filepath=data_file2)
            numpy_3d_data2 = numpy_3d_data2.astype(np.float32)

            numpy_3d_data2 = torch.Tensor(numpy_3d_data2)
            if self.transform3d is not None:
                numpy_3d_data2 = self.transform3d(numpy_3d_data2)
            numpy_3d_data2 = numpy_3d_data2.unsqueeze(0)

        if self.paths_count == 1:
            item += (numpy_3d_data1, -1)
        elif self.paths_count == 2:
            item += (numpy_3d_data1, numpy_3d_data2)

        if self.log_path is not None:
            label_local_components = self.log_data["label_local_components"][idx]
            item += (label_local_components,)

        return item


class TreesCustomDataloader3D:
    def __init__(self, args: argparse.Namespace, data_paths, log_path=None, transform2d=None, transform3d=None):
        validate_data_paths(data_paths=data_paths)
        self.args = args
        self.data_paths = data_paths
        self.log_path = log_path
        self.transform2d = transform2d
        self.transform3d = transform3d
        self._init_dataloader()

    def _init_dataloader(self):
        """
        Return Train and Val Dataloaders for the given parameters.

        Returns:
            train_dataloader: Train loader with 0.9 of the data.
            val_dataloader: Val loader with 0.1 of the data.
        """
        if self.args.dataset in V1_3D_DATASETS:
            tree_dataset = TreesCustomDataset3DV1(
                data_paths=self.data_paths,
                log_path=self.log_path,
                transform2d=self.transform2d,
                transform3d=self.transform3d
            )
        elif self.args.dataset in V2_3D_DATASETS:
            tree_dataset = TreesCustomDataset3DV2(
                data_paths=self.data_paths,
                log_path=self.log_path,
                transform3d=self.transform3d
            )
        else:
            raise Exception("Dataset not available in 'Custom Dataset 3D'")

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
