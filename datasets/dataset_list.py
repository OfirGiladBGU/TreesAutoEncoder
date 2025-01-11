import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import pathlib

from datasets.custom_datasets_2d import TreesCustomDataloader2D
from datasets.custom_datasets_3d import TreesCustomDataloader3D


# Define dataset folder
DATASET_FOLDER = "parse2022" # TODO: Read from config file

# Define paths
ROOT_PATH = pathlib.Path(__file__).resolve().parent.parent
DATA_PATH = os.path.join(ROOT_PATH, "data")
DATASET_PATH = os.path.join(DATA_PATH, DATASET_FOLDER)
TRAIN_CROPPED_PATH = os.path.join(DATA_PATH, "train_cropped_data", DATASET_FOLDER)
EVAL_CROPPED_PATH = os.path.join(DATA_PATH, "eval_cropped_data", DATASET_FOLDER)
RESULTS_PATH = os.path.join(ROOT_PATH, "results", DATASET_FOLDER)

MODEL_RESULTS_PATH = os.path.join(RESULTS_PATH, "models")
PREDICT_PIPELINE_RESULTS_PATH = os.path.join(RESULTS_PATH, "predict_pipeline")
MERGE_PIPELINE_RESULTS_PATH = os.path.join(RESULTS_PATH, "merge_pipeline")
VISUALIZATION_RESULTS_PATH = os.path.join(RESULTS_PATH, "visualization")


class MNIST(object):
    def __init__(self, args):
        self.input_size = (1, 28, 28)

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
        self.input_size = (1, 28, 28)

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
        self.input_size = (1, 28, 28)

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
        self.input_size = (1, 32, 32)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1)
        ])
        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
        root = os.path.join(DATA_PATH, "cifar10")
        self.train_loader = DataLoader(
            dataset=datasets.CIFAR10(root=root, download=True, transform=transform),
            batch_size=args.batch_size,
            shuffle=False,
            **kwargs
        )
        self.test_loader = DataLoader(
            dataset=datasets.CIFAR10(root=root, train=False, transform=transform),
            batch_size=args.batch_size,
            shuffle=False,
            **kwargs
        )


# Train with 2D labels (with random holes) to predict 2D labels
class TreesDataset2DV1S(object):
    def __init__(self, args):
        self.input_size = (1, 32, 32)

        src_path = os.path.join(TRAIN_CROPPED_PATH, "labels_2d_v6")

        data_paths = [src_path]
        trees_dataloader = TreesCustomDataloader2D(data_paths=data_paths, args=args)
        self.train_loader, self.test_loader = trees_dataloader.get_dataloader()


# Train with 2D preds fixed to predict 2D labels
class TreesDataset2DV1(object):
    def __init__(self, args):
        self.input_size = (1, 32, 32)

        src_path = os.path.join(TRAIN_CROPPED_PATH, "preds_fixed_2d_v6")
        dst_path = os.path.join(TRAIN_CROPPED_PATH, "labels_2d_v6")

        data_paths = [src_path, dst_path]
        trees_dataloader = TreesCustomDataloader2D(data_paths=data_paths, args=args)
        self.train_loader, self.test_loader = trees_dataloader.get_dataloader()


# Train with 6 2D preds fixed to predict 6 2D labels
class TreesDataset2DV2(object):
    def __init__(self, args):
        self.input_size = (6, 32, 32)

        src_path = os.path.join(TRAIN_CROPPED_PATH, "preds_fixed_2d_v6")
        dst_path = os.path.join(TRAIN_CROPPED_PATH, "labels_2d_v6")

        data_paths = [src_path, dst_path]
        trees_dataloader = TreesCustomDataloader2D(data_paths=data_paths, args=args)
        self.train_loader, self.test_loader = trees_dataloader.get_dataloader()


# Train with 6 2D preds fixed to predict 6 2D labels (with regression)
class TreesDataset2DV2M(object):
    def __init__(self, args):
        self.input_size = (6, 32, 32)

        src_path = os.path.join(TRAIN_CROPPED_PATH, "preds_fixed_2d_v6")
        dst_path = os.path.join(TRAIN_CROPPED_PATH, "labels_2d_v6")
        log_path = os.path.join(TRAIN_CROPPED_PATH, "log.csv")

        data_paths = [src_path, dst_path, log_path]
        trees_dataloader = TreesCustomDataloader2D(data_paths=data_paths, args=args)
        self.train_loader, self.test_loader = trees_dataloader.get_dataloader()


# Train with 6 2D labels to predict 3D labels
class TreesDataset3DV1(object):
    def __init__(self, args):
        self.input_size = (6, 1, 32, 32)

        src_path = os.path.join(TRAIN_CROPPED_PATH, "labels_2d_v6")
        dst_path = os.path.join(TRAIN_CROPPED_PATH, "labels_3d_v6")

        data_paths = [src_path, dst_path]
        trees_dataloader = TreesCustomDataloader3D(data_paths=data_paths, args=args)
        self.train_loader, self.test_loader = trees_dataloader.get_dataloader()


# Train with 3D reconstructed labels to predict 3D labels
class TreesDataset3DV2(object):
    def __init__(self, args):
        self.input_size = (1, 32, 32, 32)

        src_path = os.path.join(TRAIN_CROPPED_PATH, "labels_3d_reconstruct_v6")
        dst_path = os.path.join(TRAIN_CROPPED_PATH, "labels_3d_v6")

        data_paths = [src_path, dst_path]
        trees_dataloader = TreesCustomDataloader3D(data_paths=data_paths, args=args)
        self.train_loader, self.test_loader = trees_dataloader.get_dataloader()


# Train with 3D reconstructed labels to predict 3D labels (with regression)
class TreesDataset3DV2M(object):
    def __init__(self, args):
        self.input_size = (1, 32, 32, 32)

        src_path = os.path.join(TRAIN_CROPPED_PATH, "labels_3d_reconstruct_v6")
        dst_path = os.path.join(TRAIN_CROPPED_PATH, "labels_3d_v6")
        log_path = os.path.join(TRAIN_CROPPED_PATH, "log.csv")

        data_paths = [src_path, dst_path, log_path]
        trees_dataloader = TreesCustomDataloader3D(data_paths=data_paths, args=args)
        self.train_loader, self.test_loader = trees_dataloader.get_dataloader()


# Train with 3D preds fixed fusion to predict 3D labels
class TreesDataset3DV3(object):
    def __init__(self, args):
        self.input_size = (1, 32, 32, 32)

        src_path = os.path.join(TRAIN_CROPPED_PATH, "preds_fixed_3d_fusion_v6")
        dst_path = os.path.join(TRAIN_CROPPED_PATH, "labels_3d_v6")

        data_paths = [src_path, dst_path]
        trees_dataloader = TreesCustomDataloader3D(data_paths=data_paths, args=args)
        self.train_loader, self.test_loader = trees_dataloader.get_dataloader()


# Train with 3D preds fixed to predict 3D labels (Direct Repair)
class TreesDataset3DV4(object):
    def __init__(self, args):
        self.input_size = (1, 32, 32, 32)

        src_path = os.path.join(TRAIN_CROPPED_PATH, "preds_fixed_3d_v6")
        dst_path = os.path.join(TRAIN_CROPPED_PATH, "labels_3d_v6")

        data_paths = [src_path, dst_path]
        trees_dataloader = TreesCustomDataloader3D(data_paths=data_paths, args=args)
        self.train_loader, self.test_loader = trees_dataloader.get_dataloader()


# Init Method
def init_dataset(args: argparse.Namespace):
    dataset_map = {
        "MNIST": MNIST,
        "EMNIST": EMNIST,
        "FashionMNIST": FashionMNIST,
        "CIFAR10": CIFAR10,
        # 2D Datasets
        "Trees2DV1S": TreesDataset2DV1S,
        "Trees2DV1": TreesDataset2DV1,
        "Trees2DV2": TreesDataset2DV2,
        "Trees2DV2M": TreesDataset2DV2M,
        # 3D Datasets
        "Trees3DV1": TreesDataset3DV1,
        "Trees3DV2": TreesDataset3DV2,
        "Trees3DV2M": TreesDataset3DV2M,
        "Trees3DV3": TreesDataset3DV3,
        "Trees3DV4": TreesDataset3DV4
    }
    if args.dataset in list(dataset_map.keys()):
        return dataset_map[args.dataset](args=args)
    else:
        raise Exception("Dataset not available in 'Dataset List'")
