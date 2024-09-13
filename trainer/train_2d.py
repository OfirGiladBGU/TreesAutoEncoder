import argparse
import random
import torch
import torch.utils.data
from torch import optim
import copy
import os
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image

from datasets.dataset_utils import apply_threshold
from trainer import loss_functions


class Trainer(object):
    def __init__(self, args: argparse.Namespace, data, model):
        self.args = args

        self.device = self.args.device
        self.data = data
        self.model = model
        self.model.to(self.device)

        # Get loaders
        self.train_loader = self.data.train_loader
        self.test_loader = self.data.test_loader

        self.datasets_for_holes = ['MNIST', 'EMNIST', 'FashionMNIST', 'CIFAR10', 'TreesV1S']

        if self.args.dataset in self.datasets_for_holes:
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        else:
            # For ae / gap_cnn
            # self.optimizer = optim.Adadelta(self.model.parameters())
            # For vgg_ae_demo / ae_v2
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def loss_function(self, out, target, original=None):
        """
        :param out: model output on the 'original' input
        :param target: the target data that the model should output
        :param original: the original input data for the model
        :return:
        """

        if self.args.dataset in ['MNIST', 'EMNIST', 'FashionMNIST']:
            out = loss_functions.reshape_inputs(input_data=out, input_size=(28 * 28,))
            target = loss_functions.reshape_inputs(input_data=target, input_size=(28 * 28,))
            LOSS = loss_functions.bce_loss(out=out, target=target, reduction='sum')

        elif self.args.dataset == 'CIFAR10':
            LOSS = loss_functions.perceptual_loss(out=out, target=target, channels=1, device=self.args.device)

        elif self.args.dataset == 'Trees2DV1S':
            out = loss_functions.reshape_inputs(input_data=out, input_size=(28 * 28,))
            target = loss_functions.reshape_inputs(input_data=target, input_size=(28 * 28,))
            LOSS = (
                0.5 * loss_functions.bce_loss(out=out, target=target, reduction='sum') +
                0.5 * loss_functions.l1_loss(out=out, target=target, reduction='sum')
            )

        elif self.args.dataset == 'Trees2DV1':
            # TODO: MIOU, Total Variation, SSIM, F1, EMD

            # ae
            # LOSS = (
            #     40 * loss_functions.reconstruction_loss(out, target) +
            #     10 * loss_functions.total_variation_lost(out, target, p=1,  device=self.args.device) +
            #     10 * loss_functions.unfilled_holes_loss(out, target, original)
            # )

            LOSS = (
                20 * loss_functions.unfilled_holes_loss(out=out, target=target, original=original) +
                10 * loss_functions.weighted_pixels_diff_loss(out=out, target=target, original=original)
            )

            # gap_cnn / ae_2d_to_2d
            # LOSS = loss_functions.mse_loss(out, target)
            # LOSS = loss_functions.mse_loss(out, target, reduction='sum')

            # LOSS = loss_functions.fill_holes_loss(out, target, original)

            # LOSS = loss_functions.perceptual_loss(out, target, device=self.args.device)

            # LOSS = (
            #     loss_functions.reconstruction_loss(out, target) +
            #     loss_functions.edge_loss(out, target, device=self.args.device)
            # )

            # LOSS = (
            #     0.5 * loss_functions.reconstruction_loss(out, target) +
            #     0.5 * loss_functions.l1_loss(out, target, reduction='sum')
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

    def create_holes(self, input_data):
        for idx in range(len(input_data)):
            white_points = torch.nonzero(input_data[idx] > 0.6)

            if white_points.size(0) > 0:
                # Randomly select one of the non-zero points
                random_point = random.choice(white_points)
                radius = random.randint(3, 5)
                self.zero_out_radius(input_data[idx], random_point, radius)

                # plt.imshow(input_data[idx].permute(1, 2, 0))
                # save_image(input_data[idx], 'img1.png')

    def _train(self, epoch):
        self.model.train()
        train_loss = 0
        for batch_idx, batch_data in enumerate(self.train_loader):
            (input_data, target_data) = batch_data

            # Notice: Faster on CPU
            if self.args.dataset in self.datasets_for_holes:
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

                self.create_holes(input_data=input_data)

            input_data = input_data.to(self.device)
            target_data = target_data.to(self.device)

            # # For Equality Check
            # res = torch.eq(input_data, target_data)
            # print(res.max())
            # print(res.min())

            self.optimizer.zero_grad()
            out_data = self.model(input_data)
            loss = self.loss_function(out=out_data, target=target_data, original=input_data)
            loss.backward()

            train_loss += loss.item()
            self.optimizer.step()
            if batch_idx % self.args.log_interval == 0:
                print('[Train] Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(input_data),
                    len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader),
                    loss.item() / len(input_data)
                ))

        print('> [Train] Epoch: {} Average loss: {:.4f}'.format(
            epoch,
            train_loss / len(self.train_loader.dataset)
        ))

    def _test(self):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.test_loader):
                (input_data, target_data) = batch_data

                if self.args.dataset in self.datasets_for_holes:
                    target_data = input_data.clone()

                    # TODO: Threshold
                    # apply_threshold(input_data, 0.5)
                    # apply_threshold(target_data, 0.5)

                    # if input_data.dtype != torch.float32:
                    #     input_data = input_data.float()
                    # if target_data.dtype != torch.float32:
                    #     target_data = target_data.float()

                    self.create_holes(input_data=input_data)

                input_data = input_data.to(self.device)
                target_data = target_data.to(self.device)

                out_data = self.model(input_data)
                test_loss += self.loss_function(out=out_data, target=target_data, original=input_data).item()

        test_loss /= len(self.test_loader.dataset)
        print('> [Test] Average Loss: {:.4f}'.format(test_loss))

    def train(self):
        print(f"[Model: '{self.model.name}'] Training...")
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
        print(f"[Model: '{self.model.name}'] Predicting...")
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


                if self.args.dataset in self.datasets_for_holes:
                    target_images = input_images.clone()

                    # TODO: Threshold
                    # apply_threshold(input_images, 0.5)
                    # apply_threshold(target_images, 0.5)

                    # Fix for Trees dataset - Fixed problem
                    # if input_images.dtype != torch.float32:
                    #     input_images = input_images.float()
                    # if target_images.dtype != torch.float32:
                    #     target_images = target_images.float()

                    self.create_holes(input_data=input_images)

                input_images = input_images.to(self.device)
                target_images = target_images.to(self.device)

                self.model.eval()
                output_images = self.model(input_images)

                # TODO: Threshold
                # apply_threshold(output_images, 0.5)

                # Detach the images from the cuda and move them to CPU
                if self.args.cuda is True:
                    input_images = input_images.cpu()
                    target_images = target_images.cpu()
                    output_images = output_images.cpu()

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
