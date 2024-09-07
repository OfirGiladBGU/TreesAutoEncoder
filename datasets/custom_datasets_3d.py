import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import cv2
import pathlib

from datasets.dataset_utils import convert_nii_gz_to_numpy


class TreesCustomDataset3DV1(Dataset):
    def __init__(self, data_paths: list, transform2d=None, transform3d=None):
        self.data_paths = data_paths
        self.transform2d = transform2d
        self.transform3d = transform3d
        self.to_tensor = transforms.ToTensor()

        self.paths_count = len(data_paths)
        if self.paths_count == 1:
            self.data_files1 = pathlib.Path(data_paths[0]).rglob("*.png")
            self.data_files1 = sorted(self.data_files1)
        elif self.paths_count == 2:
            self.data_files1 = pathlib.Path(data_paths[0]).rglob("*.png")
            self.data_files1 = sorted(self.data_files1)

            self.data_files2 = pathlib.Path(data_paths[1]).rglob("*.nii.gz")
            self.data_files2 = sorted(self.data_files2)
        else:
            raise ValueError("Invalid number of data paths")

        self.scans_count = len(self.data_files2)

    def __len__(self):
        return self.scans_count

    def __getitem__(self, idx):
        data_idx = idx * 6

        batch = list()
        for i in range(6):
            data_file1 = str(self.data_files1[data_idx + i])
            numpy_2d_data1 = cv2.imread(data_file1)
            numpy_2d_data1 = cv2.cvtColor(numpy_2d_data1, cv2.COLOR_BGR2GRAY)

            numpy_2d_data1 = self.to_tensor(numpy_2d_data1)
            if self.transform2d is not None:
                numpy_2d_data1 = self.transform2d(numpy_2d_data1)

            batch.append(numpy_2d_data1)

        # Only 6 images
        if self.paths_count == 1:
            target = -1

        # 6 images + 1 3D data
        elif self.paths_count == 2:
            data_file2 =  str(self.data_files2[idx])
            numpy_3d_data2 = convert_nii_gz_to_numpy(data_filepath=data_file2)
            numpy_3d_data2 = numpy_3d_data2.astype(np.float32)

            numpy_3d_data2 = torch.Tensor(numpy_3d_data2)
            if self.transform3d is not None:
                numpy_3d_data2 = self.transform3d(numpy_3d_data2)

            target = numpy_3d_data2

        # Invalid number of data paths
        else:
            raise ValueError("Invalid number of data paths")

        batch = torch.stack(batch)
        target = torch.stack([target])
        return batch, target


class TreesCustomDataset3DV2(Dataset):
    def __init__(self, data_paths: list, transform3d=None):
        self.data_paths = data_paths
        self.transform3d = transform3d

        self.paths_count = len(data_paths)
        if self.paths_count == 1:
            self.data_files1 = pathlib.Path(data_paths[0]).rglob("*.nii.gz")
            self.data_files1 = sorted(self.data_files1)

        elif self.paths_count == 2:
            self.data_files1 = pathlib.Path(data_paths[0]).rglob("*.nii.gz")
            self.data_files1 = sorted(self.data_files1)

            self.data_files2 = pathlib.Path(data_paths[1]).rglob("*.nii.gz")
            self.data_files2 = sorted(self.data_files2)

        else:
            raise ValueError("Invalid number of data paths")

        self.scans_count = len(self.data_files2)

    def __len__(self):
        return self.scans_count

    def __getitem__(self, idx):
        data_file1 = str(self.data_files1[idx])
        numpy_3d_data1 = convert_nii_gz_to_numpy(data_filepath=data_file1)
        numpy_3d_data1 = numpy_3d_data1.astype(np.float32)

        numpy_3d_data1 = torch.Tensor(numpy_3d_data1)
        if self.transform3d is not None:
            numpy_3d_data1 = self.transform3d(numpy_3d_data1)

        # Only 1 3D data
        if self.paths_count == 1:
            return numpy_3d_data1, -1

        # 1 3D input + 1 3D target
        elif self.paths_count == 2:
            data_file2 = str(self.data_files2[idx])
            numpy_3d_data2 = convert_nii_gz_to_numpy(data_filepath=data_file2)
            numpy_3d_data2 = numpy_3d_data2.astype(np.float32)

            numpy_3d_data2 = torch.Tensor(numpy_3d_data2)
            if self.transform3d is not None:
                numpy_3d_data2 = self.transform3d(numpy_3d_data2)

            numpy_3d_data1 = numpy_3d_data1.unsqueeze(0)
            numpy_3d_data2 = numpy_3d_data2.unsqueeze(0)
            return numpy_3d_data1, numpy_3d_data2


class TreesCustomDataloader3D:
    def __init__(self, data_paths, args, transform2d=None, transform3d=None):
        self.data_paths = data_paths
        self.args = args
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
        if self.args.dataset == 'Trees3DV1':
            tree_dataset = TreesCustomDataset3DV1(self.data_paths, self.transform2d, self.transform3d)
        elif self.args.dataset == 'Trees3DV2':
            tree_dataset = TreesCustomDataset3DV2(self.data_paths, self.transform3d)
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
        if self.args is not None:
            kwargs = {'num_workers': 1, 'pin_memory': True} if self.args.cuda else {}
            self.train_dataloader = DataLoader(
                dataset=train_data,
                batch_size=self.args.batch_size,
                shuffle=True,
                **kwargs
            )
            self.test_dataloader = DataLoader(
                dataset=test_data,
                batch_size=self.args.batch_size,
                shuffle=False,
                **kwargs
            )
        else:
            self.train_dataloader = DataLoader(
                dataset=train_data,
                batch_size=1,
                shuffle=True
            )
            self.test_dataloader = DataLoader(
                dataset=test_data,
                batch_size=1,
                shuffle=False
            )

    def get_dataloader(self):
        return self.train_dataloader, self.test_dataloader
