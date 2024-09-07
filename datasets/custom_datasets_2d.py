import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import cv2
import pathlib


class TreesCustomDatasetV1(Dataset):
    def __init__(self, data_paths: list, transform=None):
        self.data_paths = data_paths
        self.transform = transform
        self.to_tensor = transforms.ToTensor()

        self.paths_count = len(data_paths)
        if self.paths_count == 1:
            self.data_files1 = pathlib.Path(data_paths[0]).rglob("*.png")
            self.data_files1 = sorted(self.data_files1)

        elif self.paths_count == 2:
            self.data_files1 = pathlib.Path(data_paths[0]).rglob("*.png")
            self.data_files1 = sorted(self.data_files1)

            self.data_files2 = pathlib.Path(data_paths[1]).rglob("*.png")
            self.data_files2 = sorted(self.data_files2)

        else:
            raise ValueError("Invalid number of data paths")

        self.dataset_count = len(self.data_files1)

    def __len__(self):
        return self.dataset_count

    def __getitem__(self, idx):
        data_file1 = str(self.data_files1[idx])
        numpy_2d_data1 = cv2.imread(data_file1)
        numpy_2d_data1 = cv2.cvtColor(numpy_2d_data1, cv2.COLOR_BGR2GRAY)

        numpy_2d_data1 = self.to_tensor(numpy_2d_data1)
        if self.transform is not None:
            numpy_2d_data1 = self.transform(numpy_2d_data1)

        if self.paths_count == 1:
            return numpy_2d_data1, -1

        elif self.paths_count == 2:
            data_file2 = str(self.data_files2[idx])
            numpy_2d_data2 = cv2.imread(data_file2)
            numpy_2d_data2 = cv2.cvtColor(numpy_2d_data2, cv2.COLOR_BGR2GRAY)

            numpy_2d_data2 = self.to_tensor(numpy_2d_data2)
            if self.transform is not None:
                numpy_2d_data2 = self.transform(numpy_2d_data2)

            return numpy_2d_data1, numpy_2d_data2


# TODO: Check if will be useful
class TreesCustomDatasetV2(Dataset):
    def __init__(self, data_paths: list, transform=None):
        self.data_paths = data_paths
        self.transform = transform
        self.to_tensor = transforms.ToTensor()

        self.paths_count = len(data_paths)
        if self.paths_count == 1:
            self.data_files1 = pathlib.Path(data_paths[0]).rglob("*.png")
            self.data_files1 = sorted(self.data_files1)

        elif self.paths_count == 2:
            self.data_files1 = pathlib.Path(data_paths[0]).rglob("*.png")
            self.data_files1 = sorted(self.data_files1)

            self.data_files2 = pathlib.Path(data_paths[1]).rglob("*.png")
            self.data_files2 = sorted(self.data_files2)

        else:
            raise ValueError("Invalid number of data paths")

        self.dataset_count = int(len(self.data_files1) / 6)

    def __len__(self):
        return self.dataset_count

    def __getitem__(self, idx):
        data_idx = idx * 6

        batch1 = list()
        batch2 = list()
        for i in range(6):
            data_file1 = str(self.data_files1[data_idx + i])
            numpy_2d_data1 = cv2.imread(data_file1)
            numpy_2d_data1 = cv2.cvtColor(numpy_2d_data1, cv2.COLOR_BGR2GRAY)

            numpy_2d_data1 = self.to_tensor(numpy_2d_data1)
            if self.transform is not None:
                numpy_2d_data1 = self.transform(numpy_2d_data1)

            if self.paths_count == 1:
                batch1.append(numpy_2d_data1)
                batch2.append(-1)

            elif self.paths_count == 2:
                data_file2 = str(self.data_files2[data_idx + i])
                numpy_2d_data2 = cv2.imread(data_file2)
                numpy_2d_data2 = cv2.cvtColor(numpy_2d_data2, cv2.COLOR_BGR2GRAY)

                numpy_2d_data2 = self.to_tensor(numpy_2d_data2)
                if self.transform is not None:
                    numpy_2d_data2 = self.transform(numpy_2d_data2)

                batch1.append(numpy_2d_data1)
                batch2.append(numpy_2d_data2)

        return batch1, batch2


class TreesCustomDataloader:
    def __init__(self, data_paths, args, transform=None):
        self.data_paths = data_paths
        self.args = args
        self.transform = transform
        self._init_dataloader()

    def _init_dataloader(self):
        """
        Return Train and Val Dataloaders for the given parameters.

        Returns:
            train_dataloader: Train loader with 0.9 of the data.
            val_dataloader: Val loader with 0.1 of the data.
        """
        if self.args.dataset == 'TreesV1':
            tree_dataset = TreesCustomDatasetV1(self.data_paths, self.transform)
        elif self.args.dataset == 'TreesV2':
            tree_dataset = TreesCustomDatasetV2(self.data_paths, self.transform)
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
                shuffle=True,
            )
            self.test_dataloader = DataLoader(
                dataset=test_data,
                batch_size=1,
                shuffle=False
            )

    def get_dataloader(self):
        return self.train_dataloader, self.test_dataloader
