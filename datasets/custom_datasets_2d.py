import numpy as np
import torch
import cv2
import os
from torchvision import transforms


class TreesCustomDatasetV1(torch.utils.data.Dataset):
    def __init__(self, data_paths: list, transform=None):
        self.data_paths = data_paths
        self.transform = transform
        self.to_tensor = transforms.ToTensor()

        self.paths_count = len(data_paths)
        if self.paths_count == 1:
            self.data_files1 = os.listdir(data_paths[0])
            self.data_files1.sort()
        elif self.paths_count == 2:
            self.data_files1 = os.listdir(data_paths[0])
            self.data_files1.sort()
            self.data_files2 = os.listdir(data_paths[1])
            self.data_files2.sort()
        else:
            raise ValueError("Invalid number of data paths")

        self.scans_count = len(self.data_files1)

    def __len__(self):
        return self.scans_count

    def __getitem__(self, idx):
        data_file1 = os.path.join(self.data_paths[0], self.data_files1[idx])
        image_numpy1 = cv2.imread(data_file1)
        image_numpy1 = cv2.cvtColor(image_numpy1, cv2.COLOR_BGR2GRAY)
        # image_numpy1 = np.expand_dims(image_numpy1, axis=0)
        # image_numpy1.astype(dtype=np.float32)

        # if image_numpy1.max() > 0:
        #     image_numpy1 = image_numpy1 / image_numpy1.max()

        image_numpy1 = self.to_tensor(image_numpy1)
        if self.transform is not None:
            image_numpy1 = self.transform(image_numpy1)

        if self.paths_count == 1:
            return image_numpy1, -1

        elif self.paths_count == 2:
            data_file2 = os.path.join(self.data_paths[1], self.data_files2[idx])
            image_numpy2 = cv2.imread(data_file2)
            image_numpy2 = cv2.cvtColor(image_numpy2, cv2.COLOR_BGR2GRAY)
            # image_numpy2 = np.expand_dims(image_numpy2, axis=0)
            # image_numpy2.astype(dtype=np.float32)

            # if image_numpy2.max() > 0:
            #     image_numpy2 = image_numpy2 / image_numpy2.max()

            image_numpy2 = self.to_tensor(image_numpy2)
            if self.transform is not None:
                image_numpy2 = self.transform(image_numpy2)

            return image_numpy1, image_numpy2


# TODO: Check if will be useful
class TreesCustomDatasetV2(torch.utils.data.Dataset):
    def __init__(self, data_paths: list, transform=None):
        self.data_paths = data_paths
        self.transform = transform
        self.to_tensor = transforms.ToTensor()

        self.paths_count = len(data_paths)
        if self.paths_count == 1:
            self.data_files1 = os.listdir(data_paths[0])
            self.data_files1.sort()
        elif self.paths_count == 2:
            self.data_files1 = os.listdir(data_paths[0])
            self.data_files1.sort()
            self.data_files2 = os.listdir(data_paths[1])
            self.data_files2.sort()
        else:
            raise ValueError("Invalid number of data paths")

        self.scans_count = int(len(self.data_files1) / 6)

    def __len__(self):
        return self.scans_count

    def __getitem__(self, idx):
        data_idx = idx * 6

        batch1 = list()
        batch2 = list()
        for i in range(6):
            data_file1 = os.path.join(self.data_paths[0], self.data_files1[data_idx + i])
            image_numpy1 = cv2.imread(data_file1)
            image_numpy1 = cv2.cvtColor(image_numpy1, cv2.COLOR_BGR2GRAY)

            image_numpy1 = self.to_tensor(image_numpy1)
            if self.transform is not None:
                image_numpy1 = self.transform(image_numpy1)

            if self.paths_count == 1:
                batch1.append(image_numpy1)
                batch2.append(-1)

            elif self.paths_count == 2:
                data_file2 = os.path.join(self.data_paths[1], self.data_files2[data_idx + i])
                image_numpy2 = cv2.imread(data_file2)
                image_numpy2 = cv2.cvtColor(image_numpy2, cv2.COLOR_BGR2GRAY)

                image_numpy2 = self.to_tensor(image_numpy2)
                if self.transform is not None:
                    image_numpy2 = self.transform(image_numpy2)

                batch1.append(image_numpy1)
                batch2.append(image_numpy2)

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
        train_data = torch.utils.data.Subset(tree_dataset, indices=range(0, train_size))
        test_data = torch.utils.data.Subset(tree_dataset, indices=range(train_size, train_size + val_size))

        # Create dataloaders
        if self.args is not None:
            kwargs = {'num_workers': 1, 'pin_memory': True} if self.args.cuda else {}
            self.train_dataloader = torch.utils.data.DataLoader(
                dataset=train_data,
                batch_size=self.args.batch_size,
                shuffle=True,
                **kwargs
            )
            self.test_dataloader = torch.utils.data.DataLoader(
                dataset=test_data,
                batch_size=self.args.batch_size,
                shuffle=False,
                **kwargs
            )
        else:
            self.train_dataloader = torch.utils.data.DataLoader(
                dataset=train_data,
                batch_size=1,
                shuffle=True,
            )
            self.test_dataloader = torch.utils.data.DataLoader(
                dataset=test_data,
                batch_size=1,
                shuffle=False
            )

    def get_dataloader(self):
        return self.train_dataloader, self.test_dataloader
