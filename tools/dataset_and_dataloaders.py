# import numpy as np
import torch
import cv2
import os


class TreesCustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_paths: list, transform=None):
        self.data_paths = data_paths
        self.transform = transform

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

            if self.transform is not None:
                image_numpy2 = self.transform(image_numpy2)

            return image_numpy1, image_numpy2


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

        tree_dataset = TreesCustomDataset(self.data_paths, self.transform)
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
                train_data,
                batch_size=self.args.batch_size, shuffle=False, **kwargs)
            self.test_dataloader = torch.utils.data.DataLoader(
                test_data,
                batch_size=self.args.batch_size, shuffle=False, **kwargs
            )
        else:
            self.train_dataloader = torch.utils.data.DataLoader(
                train_data,
                batch_size=1, shuffle=False)
            self.test_dataloader = torch.utils.data.DataLoader(
                test_data,
                batch_size=1, shuffle=False
            )

    def get_dataloader(self):
        return self.train_dataloader, self.test_dataloader


#########
# Tests #
#########
def test1():
    data_paths = [r"./cropped_src_images"]

    tree_dataloader = TreesCustomDataloader(data_paths=data_paths, args=None)
    train_dataloader, test_dataloader = tree_dataloader.get_dataloader()

    for idx, data in enumerate(train_dataloader):
        tree_image, _ = data
        print(tree_image)
        break


def test2():
    data_paths = [r"./cropped_src_images", r"./cropped_dst_images"]

    tree_dataloader = TreesCustomDataloader(data_paths=data_paths, args=None)
    train_dataloader, test_dataloader = tree_dataloader.get_dataloader()

    for idx, data in enumerate(train_dataloader):
        tree_image1, tree_image2 = data
        print(tree_image1)
        print(tree_image2)
        break


def main():
    test1()
    test2()


if __name__ == "__main__":
    main()
