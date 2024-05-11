import torch
from torchvision import datasets, transforms
import os

from tools.dataset_and_dataloaders import TreesCustomDataloader


class MNIST(object):
    def __init__(self, args):
        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data/mnist', train=True, download=True, transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=False, **kwargs)
        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data/mnist', train=False, transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=False, **kwargs)


class EMNIST(object):
    def __init__(self, args):
        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
        self.train_loader = torch.utils.data.DataLoader(
            datasets.EMNIST('data/emnist', train=True, download=True, split='byclass', transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=False, **kwargs)
        self.test_loader = torch.utils.data.DataLoader(
            datasets.EMNIST('data/emnist', train=False, split='byclass', transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=False, **kwargs)


class FashionMNIST(object):
    def __init__(self, args):
        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
        self.train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('data/fmnist', train=True, download=True, transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=False, **kwargs)
        self.test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('data/fmnist', train=False, transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=False, **kwargs)


# Custom Dataset
class TreesDataset(object):
    def __init__(self, args):
        src_path = os.path.join(str(os.path.dirname(__file__)), "tools", "cropped_src_images")
        dst_path = os.path.join(str(os.path.dirname(__file__)), "tools", "cropped_dst_images")

        src_trees_dataloader = TreesCustomDataloader(data_path=src_path, args=args)
        dst_trees_dataloader = TreesCustomDataloader(data_path=dst_path, args=args)

        self.train_input_loader, self.test_input_loader = src_trees_dataloader.get_dataloader()
        self.train_target_loader, self.test_target_loader = dst_trees_dataloader.get_dataloader()
