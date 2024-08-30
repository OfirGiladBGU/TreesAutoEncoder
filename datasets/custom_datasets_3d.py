import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import cv2
import os
import nibabel as nib


class TreesCustomDataset3DV1(Dataset):
    def __init__(self, data_paths: list, transform2d=None, transform3d=None):
        self.data_paths = data_paths
        self.transform2d = transform2d
        self.transform3d = transform3d
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

        self.scans_count = len(self.data_files2)

    def __len__(self):
        return self.scans_count

    def __getitem__(self, idx):
        data_idx = idx * 6

        batch = list()
        for i in range(6):
            data_file1 = os.path.join(self.data_paths[0], self.data_files1[data_idx + i])
            image_numpy1 = cv2.imread(data_file1)
            image_numpy1 = cv2.cvtColor(image_numpy1, cv2.COLOR_BGR2GRAY)

            image_numpy1 = self.to_tensor(image_numpy1)
            if self.transform2d is not None:
                image_numpy1 = self.transform2d(image_numpy1)

            batch.append(image_numpy1)

        # Only 6 images
        if self.paths_count == 1:
            target = -1

        # 6 images + 1 3D data
        elif self.paths_count == 2:
            data_file2 = os.path.join(self.data_paths[1], self.data_files2[idx])
            ct_img = nib.load(data_file2)
            image_numpy2 = ct_img.get_fdata().astype(np.float32)

            image_numpy2 = torch.Tensor(image_numpy2)
            if self.transform3d is not None:
                image_numpy2 = self.transform3d(image_numpy2)

            target = image_numpy2

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
        # self.to_tensor = transforms.ToTensor()

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

        self.scans_count = len(self.data_files2)

    def __len__(self):
        return self.scans_count

    def __getitem__(self, idx):
        data_file1 = os.path.join(self.data_paths[0], self.data_files1[idx])
        ct_img1 = nib.load(data_file1)
        image_3d_numpy1 = ct_img1.get_fdata().astype(np.float32)

        image_3d_numpy1 = torch.Tensor(image_3d_numpy1)
        if self.transform3d is not None:
            image_3d_numpy1 = self.transform3d(image_3d_numpy1)

        # Only 1 3D data
        if self.paths_count == 1:
            return image_3d_numpy1, -1

        # 1 3D input + 1 3D target
        elif self.paths_count == 2:
            data_file2 = os.path.join(self.data_paths[1], self.data_files2[idx])
            ct_img2 = nib.load(data_file2)
            image_3d_numpy2 = ct_img2.get_fdata().astype(np.float32)

            image_3d_numpy2 = torch.Tensor(image_3d_numpy2)
            if self.transform3d is not None:
                image_3d_numpy2 = self.transform3d(image_3d_numpy2)

            image_3d_numpy1 = image_3d_numpy1.unsqueeze(0)
            image_3d_numpy2 = image_3d_numpy2.unsqueeze(0)
            return image_3d_numpy1, image_3d_numpy2


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
