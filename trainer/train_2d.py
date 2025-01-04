import argparse
import torch
import torch.utils.data
from torch import optim
import copy
import os
import matplotlib.pyplot as plt
import numpy as np
import wandb

from datasets.dataset_utils import apply_threshold
from datasets.custom_datasets_2d import V1_2D_DATASETS, V2_2D_DATASETS
from trainer import loss_functions
from trainer import train_utils

# TODO: remove later
import torch.nn.functional as F


class Trainer(object):
    def __init__(self, args: argparse.Namespace, dataset, model):
        self.args = args

        self.device = self.args.device
        self.dataset = dataset
        self.model = model
        self.model.to(self.device)

        # Get loaders
        self.train_loader = self.dataset.train_loader
        self.test_loader = self.dataset.test_loader

        self.datasets_for_holes = ['MNIST', 'EMNIST', 'FashionMNIST', 'CIFAR10', 'TreesV1S']

        if self.args.dataset in self.datasets_for_holes:
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        else:
            # For ae / gap_cnn
            # self.optimizer = optim.Adadelta(self.model.parameters())
            # For vgg_ae_demo / ae_v2
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        import torchvision.models as models
        # vgg = models.vgg19(pretrained=True).features[:16].eval()  # Use the first few layers of VGG19 (Legacy)
        vgg = models.vgg19().features[:16].eval()  # Use the first few layers of VGG19
        self.vgg = vgg.to(self.device)

    # TODO: Remove later
    @staticmethod
    def total_variation_loss(x):
        return torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])) + torch.sum(
            torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))

    # TODO: Remove later
    # Extract features
    def perceptual_loss(self, output, target):
        output_3ch = output.repeat(1, 3, 1, 1)
        target_3ch = target.repeat(1, 3, 1, 1)

        output_features = self.vgg(output_3ch)
        target_features = self.vgg(target_3ch)
        return F.l1_loss(output_features, target_features)

    # TODO: Remove later
    @staticmethod
    def downsample(x, scale=2):
        return F.interpolate(x, scale_factor=1 / scale, mode='bilinear', align_corners=False)

    def loss_function(self, output_data, target_data, input_data=None):
        """
        :param output_data: model output on the 'original' input
        :param target_data: the target data that the model should output
        :param input_data: the original input data for the model
        :return:
        """

        if self.args.dataset in ['MNIST', 'EMNIST', 'FashionMNIST']:
            out = loss_functions.reshape_inputs(input_data=output_data, input_size=(28 * 28,))
            target = loss_functions.reshape_inputs(input_data=target_data, input_size=(28 * 28,))
            LOSS = loss_functions.bce_loss(out=out, target=target, reduction='sum')

        elif self.args.dataset == 'CIFAR10':
            LOSS = loss_functions.perceptual_loss(out=output_data, target=target_data, channels=1, device=self.args.device)

        elif self.args.dataset == 'Trees2DV1S':
            output_data = loss_functions.reshape_inputs(input_data=output_data, input_size=(28 * 28,))
            target_data = loss_functions.reshape_inputs(input_data=target_data, input_size=(28 * 28,))
            LOSS = (
                0.5 * loss_functions.bce_loss(out=output_data, target=target_data, reduction='sum') +
                0.5 * loss_functions.l1_loss(out=output_data, target=target_data, reduction='sum')
            )

        elif self.args.dataset == 'Trees2DV1':
            # TODO: MIOU, Total Variation, SSIM, F1, EMD

            # ae
            # LOSS = (
            #     40 * loss_functions.reconstruction_loss(out, target) +
            #     10 * loss_functions.total_variation_lost(out, target, p=1,  device=self.args.device) +
            #     10 * loss_functions.unfilled_holes_loss(out, target, original)
            # )

            # Test 1
            # LOSS = (
            #     20 * loss_functions.unfilled_holes_loss(out=out, target=target, original=original) +
            #     10 * loss_functions.weighted_pixels_diff_loss(out=out, target=target, original=original)
            # )

            # Test 2
            # holes_mask = ((target_data - input_data) != 0)
            # black_mask = (target_data == 0)
            #
            # LOSS = (0.6 * F.l1_loss(output_data[holes_mask], target_data[holes_mask]) +
            #         0.2 * F.l1_loss(output_data[black_mask], target_data[black_mask]))

            # Test 3
            # holes_mask = ((target_data - input_data) > 0)
            # black_mask = (target_data == 0)
            # black_penalty = torch.where(output_data[holes_mask] < 0.001, 1.0, 0)
            #
            # LOSS = (0.6 * F.l1_loss(output_data[holes_mask], target_data[holes_mask]) +
            #         0.2 * F.l1_loss(output_data[black_mask], target_data[black_mask]) +
            #         0.2 * black_penalty.sum())

            if "confidence map" in getattr(self.model, "additional_tasks", list()):
                output_data, output_confidence_data = output_data

                # Test 4
                #
                # # Existing masks for holes and black areas
                # holes_mask = ((target_data - input_data) != 0)
                # black_mask = (target_data == 0)
                #
                # # Base L1 Loss
                # # LOSS = (0.6 * F.l1_loss(output_data[holes_mask], target_data[holes_mask]) +
                # #         0.2 * F.l1_loss(output_data[black_mask], target_data[black_mask]))
                #
                # # Add Total Variation Loss
                # # tv_loss = self.total_variation_loss(output_data)
                # # LOSS += 0.1 * tv_loss
                #
                # # Add Perceptual Loss
                # p_loss = self.perceptual_loss(output_data, target_data)
                # # LOSS += 0.5 * p_loss
                #
                # # Add Multi-Scale Loss
                # # output_low = self.downsample(output_data, scale=2)
                # # target_low = self.downsample(target_data, scale=2)
                # # multi_scale_loss = F.l1_loss(output_low, target_low)
                # # LOSS += 0.1 * multi_scale_loss
                #
                # LOSS = (0.5 * F.l1_loss(output_data[holes_mask], target_data[holes_mask]) +
                #         0.2 * F.l1_loss(output_data[black_mask], target_data[black_mask]) +
                #         0.5 * p_loss)
                #
                # # Confidence loss
                # target_confidence_data = (target_data != 0).float()
                # LOSS += F.binary_cross_entropy(output_confidence_data, target_confidence_data)

                # Test 5
                holes_mask = ((target_data - input_data) > 0)  # area that should be filled
                black_mask = (target_data == 0)  # area that should stay black

                LOSS = (0.6 * F.l1_loss(output_data[holes_mask], target_data[holes_mask]) +
                        0.4 * F.l1_loss(output_data[black_mask], target_data[black_mask]))

                # Confidence loss V2
                # target_confidence_data = (target_data != 0).float()
                # LOSS += F.binary_cross_entropy(output_confidence_data, target_confidence_data)

                # Confidence loss V3
                target_holes_confidence_data = (target_data[holes_mask] > 0).float()
                target_black_confidence_data = target_data[black_mask]

                LOSS += (0.2 * F.binary_cross_entropy(output_confidence_data[holes_mask], target_holes_confidence_data) +
                         0.8 * F.binary_cross_entropy(output_confidence_data[black_mask], target_black_confidence_data))
            else:
                # Test 4

                # Existing masks for holes and black areas
                holes_mask = ((target_data - input_data) != 0)
                black_mask = (target_data == 0)

                # Base L1 Loss
                # LOSS = (0.6 * F.l1_loss(output_data[holes_mask], target_data[holes_mask]) +
                #         0.2 * F.l1_loss(output_data[black_mask], target_data[black_mask]))

                # Add Total Variation Loss
                # tv_loss = self.total_variation_loss(output_data)
                # LOSS += 0.1 * tv_loss

                # Add Perceptual Loss
                p_loss = self.perceptual_loss(output_data, target_data)
                # LOSS += 0.5 * p_loss

                # Add Multi-Scale Loss
                # output_low = self.downsample(output_data, scale=2)
                # target_low = self.downsample(target_data, scale=2)
                # multi_scale_loss = F.l1_loss(output_low, target_low)
                # LOSS += 0.1 * multi_scale_loss

                LOSS = (0.5 * F.l1_loss(output_data[holes_mask], target_data[holes_mask]) +
                        0.2 * F.l1_loss(output_data[black_mask], target_data[black_mask]) +
                        0.5 * p_loss)

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
        elif self.args.dataset == 'Trees2DV2':
            # LOSS = (
            #     20 * loss_functions.unfilled_holes_loss(out=out, target=target, original=original) +
            #     10 * loss_functions.weighted_pixels_diff_loss(out=out, target=target, original=original)
            # )

            # target_clone = target.clone().detach()
            # original_clone = original.clone().detach()
            #
            # apply_threshold(target_clone, 0.01)
            # apply_threshold(original_clone, 0.01)
            # mask = target_clone - original_clone
            #
            # LOSS = (0.8 * mask * F.mse_loss(out, target)).sum() + 0.2 * F.l1_loss(out, target)

            # real_loss = nn.BCELoss()(discriminator(real), torch.ones_like(real))
            # fake_loss = nn.BCELoss()(discriminator(fake), torch.zeros_like(fake))
            # return real_loss + fake_loss

            # holes_mask = ((target - original) != 0)
            # non_black_mask = (target != 0)
            # LOSS = (0.6 * F.mse_loss(out[holes_mask], target[holes_mask])  +
            #         0.2 * F.mse_loss(out[non_black_mask], target[non_black_mask]) +
            #         0.2 * F.l1_loss(out, target))

            holes_mask = ((target_data - input_data) > 0)  # area that should be filled
            black_mask = (target_data == 0)  # area that should stay black
            LOSS = (0.6 * F.l1_loss(output_data[holes_mask], target_data[holes_mask]) +
                    0.4 * F.l1_loss(output_data[black_mask], target_data[black_mask]))
        else:
            raise NotImplementedError

        return LOSS

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

                train_utils.create_2d_holes(input_data=input_data)

            input_data = input_data.to(self.device)
            target_data = target_data.to(self.device)

            # # For Equality Check
            # res = torch.eq(input_data, target_data)
            # print(res.max())
            # print(res.min())

            self.optimizer.zero_grad()
            output_data = self.model(input_data)
            loss = self.loss_function(
                output_data=output_data,
                target_data=target_data,
                input_data=input_data
            )
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

        train_avg_loss = train_loss / len(self.train_loader.dataset)
        print('> [Train] Epoch: {} Average loss: {:.4f}'.format(
            epoch,
            train_avg_loss
        ))
        return train_avg_loss

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

                    train_utils.create_2d_holes(input_data=input_data)

                input_data = input_data.to(self.device)
                target_data = target_data.to(self.device)

                output_data = self.model(input_data)
                test_loss += self.loss_function(
                    output_data=output_data,
                    target_data=target_data,
                    input_data=input_data
                ).item()

        test_avg_loss = test_loss / len(self.test_loader.dataset)
        print('> [Test] Average Loss: {:.4f}'.format(test_avg_loss))
        return test_avg_loss

    def train(self, use_weights=False):
        if use_weights is True:
            print("Loading Model Weights")
            self.model.load_state_dict(torch.load(self.args.weights_filepath))

        print(f"[Model: '{self.model.model_name}'] Training...")
        try:
            for epoch in range(1, self.args.epochs + 1):
                train_avg_loss = self._train(epoch=epoch)
                test_avg_loss = self._test()
                wandb.log(
                    data={"Train Loss": train_avg_loss, "Test Loss": test_avg_loss},
                    step=epoch
                )
        except (KeyboardInterrupt, SystemExit):
            print("Manual Interruption")

        print("Saving Model Weights")
        model_parameters = copy.deepcopy(self.model.state_dict())
        torch.save(model_parameters, self.args.weights_filepath)

    # TODO: Handle V1 and V2 cases
    def predict(self):
        print(f"[Model: '{self.model.model_name}'] Predicting...")
        os.makedirs(name=self.args.results_path, exist_ok=True)

        # Load model weights
        if os.path.exists(self.args.weights_filepath):
            self.model.load_state_dict(torch.load(self.args.weights_filepath))

        with torch.no_grad():
            batches_to_plot = 4
            for batch_idx in range(batches_to_plot):
                # Get the images from the test loader
                batch_num = batch_idx + 1
                data = iter(self.test_loader)
                for _ in range(batch_num):
                    input_data, target_data = next(data)

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

                    train_utils.create_2d_holes(input_data=input_data)

                input_data = input_data.to(self.device)
                target_data = target_data.to(self.device)

                self.model.eval()
                output_data = self.model(input_data)

                if "confidence map" in getattr(self.model, "additional_tasks", list()):
                    output_data, output_confidence_data = output_data
                    output_data = torch.where(output_confidence_data > 0.5, output_data, 0)

                # TODO: Threshold
                # apply_threshold(output_images, 0.5)

                fusion_data = input_data + torch.where(input_data == 0, output_data, 0)
                # fusion_data = torch.where(input_data == 0, output_data, 0)

                # Detach the images from the cuda and move them to CPU
                if self.args.cuda is True:
                    input_data = input_data.cpu()
                    target_data = target_data.cpu()
                    output_data = output_data.cpu()
                    fusion_data = fusion_data.cpu()

                # Convert (b, 6, w, h) to (6*b, 1, w, h) - Trees2DV2
                if input_data.shape[1] == 6:
                    x, y = self.args.input_size[1:]
                    input_data = input_data.view(-1, 1, x, y)
                    target_data = target_data.view(-1, 1, x, y)
                    output_data = output_data.view(-1, 1, x, y)
                    fusion_data = fusion_data.view(-1, 1, x, y)

                #################
                # Visualization #
                #################

                # Create a grid of images
                columns = 4
                rows = input_data.shape[0]
                fig = plt.figure(figsize=(columns + 0.5, rows + 0.5))
                ax = []
                for i in range(rows):
                    j = 0

                    # Input
                    j += 1
                    ax.append(fig.add_subplot(rows, columns, i * columns + j))
                    numpy_image = input_data[i].numpy()
                    plt.imshow(np.transpose(numpy_image, (1, 2, 0)), cmap='gray')

                    # Target
                    j += 1
                    ax.append(fig.add_subplot(rows, columns, i * columns + j))
                    numpy_image = target_data[i].numpy()
                    plt.imshow(np.transpose(numpy_image, (1, 2, 0)), cmap='gray')

                    # Output
                    j += 1
                    ax.append(fig.add_subplot(rows, columns, i * columns + j))
                    numpy_image = output_data[i].numpy()
                    plt.imshow(np.transpose(numpy_image, (1, 2, 0)), cmap='gray')

                    # Fusion
                    j += 1
                    ax.append(fig.add_subplot(rows, columns, i * columns + j))
                    numpy_image = fusion_data[i].numpy()
                    plt.imshow(np.transpose(numpy_image, (1, 2, 0)), cmap='gray')

                ax[0].set_title("Input:")
                ax[1].set_title("Target:")
                ax[2].set_title("Output:")
                ax[3].set_title("Fusion:")
                fig.tight_layout()
                save_filename = os.path.join(self.args.results_path, f"{self.args.dataset}_{batch_num}.png")
                plt.savefig(save_filename)
                wandb.log(
                    data={f"Batch {batch_num} - Predict Plots": wandb.Image(plt)}
                )
