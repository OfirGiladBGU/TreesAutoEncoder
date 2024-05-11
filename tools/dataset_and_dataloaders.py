import os
import torch
import cv2


class TreesCustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.data_files = os.listdir(data_path)
        self.data_files.sort()
        self.scans_count = len(self.data_files)

    def __len__(self):
        return self.scans_count

    def __getitem__(self, idx):
        data_file = os.path.join(self.data_path, self.data_files[idx])
        image_numpy = cv2.imread(data_file)
        image_numpy = cv2.cvtColor(image_numpy, cv2.COLOR_BGR2GRAY)

        if self.transform:
            image_numpy = self.transform(image_numpy)

        return image_numpy, -1


class TreesCustomDataloader:
    def __init__(self, data_path, args, transform=None):
        self.data_path = data_path
        self.args = args
        self.transform = transform
        self._init_dataloader()

    def _init_dataloader(self):
        """
        Return Train and Val Dataloaders for the given parameters.

        Returns:
            train_dataloader: Train loader with 0.7 of the data.
            val_dataloader: Val loader with 0.3 of the data.
        """

        tree_dataset = TreesCustomDataset(self.data_path, self.transform)
        dataset_size = len(tree_dataset)
        train_size = int(dataset_size * 0.7)
        val_size = dataset_size - train_size

        train_data, test_data = torch.utils.data.random_split(tree_dataset, [train_size, val_size])

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


def main():
    data_path = r"./cropped_src_images"

    tree_dataloader = TreesCustomDataloader(data_path=data_path, args=None)
    train_dataloader, test_dataloader = tree_dataloader.get_dataloader()

    for idx, data in enumerate(train_dataloader):
        tree_image, _ = data
        print(tree_image)
        break


if __name__ == "__main__":
    main()
