import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import pathlib

from datasets.custom_datasets_2d import TreesCustomDataloader
from datasets.custom_datasets_3d import TreesCustomDataloader3D

ROOT_PATH = pathlib.Path(__file__).resolve().parent.parent
DATA_PATH = os.path.join(ROOT_PATH, "data")
DATASET_PATH = os.path.join(DATA_PATH, "parse2022")  # TODO: Read from config file
CROPPED_PATH = os.path.join(DATA_PATH, "cropped_data")


class MNIST(object):
    def __init__(self, args):
        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
        root = os.path.join(DATA_PATH, "mnist")
        self.train_loader = DataLoader(
            dataset=datasets.MNIST(root=root, train=True, download=True, transform=transforms.ToTensor()),
            batch_size=args.batch_size,
            shuffle=False,
            **kwargs
        )
        self.test_loader = DataLoader(
            dataset=datasets.MNIST(root=root, train=False, transform=transforms.ToTensor()),
            batch_size=args.batch_size,
            shuffle=False,
            **kwargs
        )


class EMNIST(object):
    def __init__(self, args):
        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
        root = os.path.join(DATA_PATH, "emnist")
        self.train_loader = DataLoader(
            dataset=datasets.EMNIST(root=root, train=True, download=True, split='byclass', transform=transforms.ToTensor()),
            batch_size=args.batch_size,
            shuffle=False,
            **kwargs
        )
        self.test_loader = DataLoader(
            dataset=datasets.EMNIST(root=root, train=False, split='byclass', transform=transforms.ToTensor()),
            batch_size=args.batch_size,
            shuffle=False,
            **kwargs
        )


class FashionMNIST(object):
    def __init__(self, args):
        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
        root = os.path.join(DATA_PATH, "fmnist")
        self.train_loader = DataLoader(
            dataset=datasets.FashionMNIST(root=root, train=True, download=True, transform=transforms.ToTensor()),
            batch_size=args.batch_size,
            shuffle=False,
            **kwargs
        )
        self.test_loader = DataLoader(
            dataset=datasets.FashionMNIST(root=root, train=False, transform=transforms.ToTensor()),
            batch_size=args.batch_size,
            shuffle=False,
            **kwargs
        )


class CIFAR10(object):
    def __init__(self, args):
        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
        root = os.path.join(DATA_PATH, "cifar10")
        self.train_loader = DataLoader(
            dataset=datasets.CIFAR10(root=root, download=True, transform=transforms.ToTensor()),
            batch_size=args.batch_size,
            shuffle=False,
            **kwargs
        )
        self.test_loader = DataLoader(
            dataset=datasets.CIFAR10(root=root, train=False, transform=transforms.ToTensor()),
            batch_size=args.batch_size,
            shuffle=False,
            **kwargs
        )


# Custom Dataset
class TreesDatasetV1(object):
    def __init__(self, args):
        src_path = os.path.join(CROPPED_PATH, "parse_preds_mini_cropped_v5")
        dst_path = os.path.join(CROPPED_PATH, "parse_labels_mini_cropped_v5")

        data_paths = [src_path, dst_path]
        trees_dataloader = TreesCustomDataloader(data_paths=data_paths, args=args)
        self.train_loader, self.test_loader = trees_dataloader.get_dataloader()


class TreesDatasetV2(object):
    def __init__(self, args):
        src_path = os.path.join(CROPPED_PATH, "mini_cropped_images")

        data_paths = [src_path]
        trees_dataloader = TreesCustomDataloader(data_paths=data_paths, args=args)
        self.train_loader, self.test_loader = trees_dataloader.get_dataloader()


class TreesDataset3DV1(object):
    def __init__(self, args):
        src_path = os.path.join(CROPPED_PATH, "parse_labels_mini_cropped_v5")
        dst_path = os.path.join(CROPPED_PATH, "parse_labels_mini_cropped_3d_v5")

        data_paths = [src_path, dst_path]
        trees_dataloader = TreesCustomDataloader3D(data_paths=data_paths, args=args)
        self.train_loader, self.test_loader = trees_dataloader.get_dataloader()


class TreesDataset3DV2(object):
    def __init__(self, args):
        src_path = os.path.join(CROPPED_PATH, "parse_labels_mini_cropped_3d_reconstruct_v5")
        dst_path = os.path.join(CROPPED_PATH, "parse_labels_mini_cropped_3d_v5")

        data_paths = [src_path, dst_path]
        trees_dataloader = TreesCustomDataloader3D(data_paths=data_paths, args=args)
        self.train_loader, self.test_loader = trees_dataloader.get_dataloader()
