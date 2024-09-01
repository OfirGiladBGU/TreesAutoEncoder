import argparse
import random
import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
import copy
import os
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image

from datasets.dataset_utils import apply_threshold
from datasets.dataset_list import MNIST, EMNIST, FashionMNIST, CIFAR10, TreesDatasetV0, TreesDatasetV1, TreesDatasetV2
from trainer import loss_functions


class Trainer(object):
    def __init__(self, args: argparse.Namespace, model):
        self.args = args
        self.device = args.device
        self._init_dataset()

        # Get loaders
        self.train_loader = self.data.train_loader
        self.test_loader = self.data.test_loader

        self.model = model
        self.model.to(self.device)
        self.input_size = model.input_size

        if args.dataset in ['MNIST', 'EMNIST', 'FashionMNIST']:
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        elif args.dataset == 'CIFAR10':
            self.optimizer = optim.Adam(self.model.parameters())
        else:
            # For ae / gap_cnn
            # self.optimizer = optim.Adadelta(self.model.parameters())
            # For vgg_ae_demo / ae_v2
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def _init_dataset(self):
        if self.args.dataset == 'MNIST':
            self.data = MNIST(self.args)
        elif self.args.dataset == 'EMNIST':
            self.data = EMNIST(self.args)
        elif self.args.dataset == 'FashionMNIST':
            self.data = FashionMNIST(self.args)
        elif self.args.dataset == 'CIFAR10':
            self.data = CIFAR10(self.args)
        # Custom Dataset
        elif self.args.dataset == 'TreesV0':
            self.data = TreesDatasetV0(self.args)
        elif self.args.dataset == 'TreesV1':
            self.data = TreesDatasetV1(self.args)
        elif self.args.dataset == 'TreesV2':
            self.data = TreesDatasetV2(self.args)
        else:
            raise Exception("Dataset not supported")

    def loss_function(self, out, target, original=None):
        """
        :param out: model output on the 'original' input
        :param target: the target data that the model should output
        :param original: the original input data for the model
        :return:
        """

        if self.args.dataset in ['MNIST', 'EMNIST', 'FashionMNIST']:
            out, target = loss_functions.reshape_inputs(out, target, input_size=(28 * 28, ))
            LOSS = F.binary_cross_entropy(out, target, reduction='sum')

        elif self.args.dataset == 'CIFAR10':
            LOSS = loss_functions.perceptual_loss(out, target, channels=3, device=self.args.device)

        elif self.args.dataset == 'TreesV0':
            out, target = loss_functions.reshape_inputs(out, target, input_size=(28 * 28, ))
            LOSS = (
                0.5 * F.binary_cross_entropy(out, target, reduction='sum') +
                0.5 * F.l1_loss(out, target, reduction='sum')
            )

        elif self.args.dataset == 'TreesV1':
            # TODO: MIOU, Total Variation, SSIM, F1, EMD

            # ae
            # LOSS = (
            #     40 * loss_functions.reconstruction_loss(out, target) +
            #     10 * loss_functions.total_variation_lost(out, target, p=1,  device=self.args.device) +
            #     10 * loss_functions.unfilled_holes_loss(out, target, original)
            # )

            LOSS = (
                20 * loss_functions.unfilled_holes_loss(out, target, original) +
                10 * loss_functions.weighted_pixels_diff_loss(out, target, original)
            )

            # gap_cnn / ae_2d_to_2d
            # LOSS = F.mse_loss(out, target)
            # LOSS = F.mse_loss(out, target, reduction='sum')

            # LOSS = loss_functions.fill_holes_loss(out, target, original)

            # LOSS = loss_functions.perceptual_loss(out, target, device=self.args.device)

            # LOSS = (
            #     loss_functions.reconstruction_loss(out, target) +
            #     loss_functions.edge_loss(out, target, device=self.args.device)
            # )

            # LOSS = (
            #     0.5 *  loss_functions.reconstruction_loss(out, target) +
            #     0.5 * F.l1_loss(out, target, reduction='sum')
            # )
        else:
            raise NotImplementedError

        return LOSS

    @staticmethod
    def zero_out_radius(tensor, point, radius):
        x, y = point[1], point[2]  # Get the coordinates
        for i in range(max(0, x - radius), min(tensor.size(1), x + radius + 1)):
            for j in range(max(0, y - radius), min(tensor.size(2), y + radius + 1)):
                if (i - x) ** 2 + (j - y) ** 2 <= radius ** 2:
                    tensor[0, i, j] = 0

    def create_holes(self, target_data):
        for idx in range(len(target_data)):
            white_points = torch.nonzero(target_data[idx] > 0.6)

            if white_points.size(0) > 0:
                # Randomly select one of the non-zero points
                random_point = random.choice(white_points)
                radius = random.randint(3, 5)
                self.zero_out_radius(target_data[idx], random_point, radius)

                # plt.imshow(target_data[idx].permute(1, 2, 0))
                # save_image(target_data[idx], 'img1.png')

    def _train(self, epoch):
        self.model.train()
        train_loss = 0
        for batch_idx, (input_data, target_data) in enumerate(self.train_loader):
            if self.args.dataset not in ['TreesV1', 'TreesV2']:
                target_data = input_data.clone()

            # TODO: Threshold
            # apply_threshold(input_data, 0.5)
            # apply_threshold(target_data, 0.5)

            # Fix for Trees dataset - Fixed problem
            # if input_data.dtype != torch.float32:
            #     input_data = input_data.float()
            # if target_data.dtype != torch.float32:
            #     target_data = target_data.float()

            # # For Equality Check
            # res = torch.eq(input_data, target_data)
            # print(res.max())
            # print(res.min())

            # Notice: Faster on CPU
            if self.args.dataset not in ['TreesV1', 'TreesV2', 'CIFAR10']:
                self.create_holes(input_data)

            input_data = input_data.to(self.device)
            target_data = target_data.to(self.device)

            # # For Equality Check
            # res = torch.eq(input_data, target_data)
            # print(res.max())
            # print(res.min())

            self.optimizer.zero_grad()
            recon_batch = self.model(input_data)
            loss = self.loss_function(out=recon_batch, target=target_data, original=input_data)
            loss.backward()

            train_loss += loss.item()
            self.optimizer.step()
            if batch_idx % self.args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(input_data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader),
                    loss.item() / len(input_data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / len(self.train_loader.dataset)))

    def _test(self):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, (input_data, target_data) in enumerate(self.test_loader):
                input_data = input_data.to(self.device)
                if self.args.dataset != 'TreesV1':
                    target_data = input_data.clone()
                target_data = target_data.to(self.device)

                # TODO: Threshold
                # apply_threshold(input_data, 0.5)
                # apply_threshold(target_data, 0.5)

                # if input_data.dtype != torch.float32:
                #     input_data = input_data.float()
                # if target_data.dtype != torch.float32:
                #     target_data = target_data.float()

                if self.args.dataset not in ['TreesV1', 'TreesV2', 'CIFAR10']:
                    self.create_holes(input_data)

                recon_batch = self.model(input_data)
                test_loss += self.loss_function(out=recon_batch, target=target_data, original=input_data).item()

        test_loss /= len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))

    def train(self):
        try:
            for epoch in range(1, self.args.epochs + 1):
                self._train(epoch=epoch)
                self._test()
        except (KeyboardInterrupt, SystemExit):
            print("Manual Interruption")

        print("Saving Model Weights")
        model_parameters = copy.deepcopy(self.model.state_dict())
        torch.save(model_parameters, self.args.weights_filepath)

    def predict(self):
        print("Predicting")
        os.makedirs(name=self.args.results_path, exist_ok=True)

        # Load model weights
        if os.path.exists(self.args.weights_filepath):
            self.model.load_state_dict(torch.load(self.args.weights_filepath))

        with torch.no_grad():
            for b in range(4):
                # Get the images from the test loader
                batch_num = b + 1
                data = iter(self.test_loader)
                for i in range(batch_num):
                    input_images, target_images = next(data)
                input_images = input_images.to(self.device)
                if self.args.dataset != 'TreesV1':
                    target_images = input_images.clone().detach()
                target_images = target_images.to(self.device)

                # TODO: Threshold
                # apply_threshold(input_images, 0.5)
                # apply_threshold(target_images, 0.5)

                # Fix for Trees dataset - Fixed problem
                # if input_images.dtype != torch.float32:
                #     input_images = input_images.float()
                # if target_images.dtype != torch.float32:
                #     target_images = target_images.float()

                # Create holes in the input images
                if self.args.dataset not in ['TreesV1', 'TreesV2', 'CIFAR10']:
                    self.create_holes(input_images)

                self.model.eval()
                output_images = self.model(input_images)

                # TODO: Threshold
                # apply_threshold(output_images, 0.5)

                # Detach the images from the cuda and move them to CPU
                if self.args.cuda:
                    input_images = input_images.cpu().detach()
                    target_images = target_images.cpu().detach()
                    output_images = output_images.cpu().detach()

                # Create a grid of images
                columns = 3
                rows = 25
                fig = plt.figure(figsize=(columns + 0.5, rows + 0.5))
                ax = []
                for i in range(rows):
                    # Input
                    ax.append(fig.add_subplot(rows, columns, i * columns + 1))
                    npimg = input_images[i].numpy()
                    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')

                    # Target
                    ax.append(fig.add_subplot(rows, columns, i * columns + 2))
                    npimg = target_images[i].numpy()
                    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')

                    # Output
                    ax.append(fig.add_subplot(rows, columns, i * columns + 3))
                    npimg = output_images[i].numpy()
                    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')

                ax[0].set_title("Input:")
                ax[1].set_title("Target:")
                ax[2].set_title("Output:")
                fig.tight_layout()
                plt.savefig(os.path.join(
                    self.args.results_path,
                    f'output_{self.args.dataset}_{self.model.model_name}_{b + 1}.png')
                )
