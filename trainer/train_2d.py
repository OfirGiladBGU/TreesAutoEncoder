import argparse
import torch
import torch.utils.data
from torch import optim
import copy
import os
import matplotlib.pyplot as plt
import numpy as np
import wandb
from tqdm import tqdm

from datasets.dataset_configurations import V1_2D_DATASETS, V2_2D_DATASETS
from datasets.dataset_utils import apply_threshold
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
        # Handle additional_tasks
        if "confidence map" in getattr(self.model, "additional_tasks", list()):
            output_data, output_confidence_data = output_data
        else:
            output_confidence_data = None


        if self.args.dataset in ['MNIST', 'EMNIST', 'FashionMNIST']:
            # Test 1
            # LOSS = loss_functions.bce_loss(out=output_data, target=target_data, reduction='sum')

            # Test 2
            # holes_mask = ((target_data > 0) & (input_data == 0)).float()  # Convert to float for multiplication
            # black_mask = (target_data == 0)  # area that should stay black
            #
            # diff = torch.abs(output_data - target_data)
            # masked_diff = diff * (holes_mask + black_mask)
            #
            # # Normalize by the number of pixels in the mask
            # LOSS = masked_diff.sum() / holes_mask.sum()
            # LOSS += loss_functions.perceptual_loss(out=output_data, target=target_data, channels=1, device=self.args.device)
            # LOSS += loss_functions.bce_dice_loss(out=output_data, target=target_data)

            # Test 3
            # holes_mask = ((target_data - input_data) > 0)  # area that should be filled
            # black_mask = (target_data == 0)  # area that should stay black
            #
            # LOSS = (0.6 * F.l1_loss(output_data[holes_mask], target_data[holes_mask]) +
            #         0.4 * F.l1_loss(output_data[black_mask], target_data[black_mask]))

            # Test 4
            keep_mask1 = (input_data > 0).float()  # Area that should stay unchanged
            black_mask1 = (target_data == 0).float()  # Area that should stay black
            fill_mask1 = ((target_data > 0) & (input_data == 0)).float()  # Area that should be filled
            weighted_mask1 = 80.0 * fill_mask1 + 15.0 * keep_mask1 + 5.0 * black_mask1

            output_data_1 = output_data * weighted_mask1
            target_data_1 = target_data * weighted_mask1

            # Normalize by the number of pixels in the mask
            LOSS = loss_functions.l1_loss(out=output_data_1, target=target_data_1, reduction='sum')
            LOSS += loss_functions.perceptual_loss(out=output_data, target=target_data, channels=1, device=self.args.device)
            LOSS += loss_functions.bce_dice_loss(out=output_data, target=target_data)

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


            # Test 5
            # holes_mask = ((target_data - input_data) > 0)  # area that should be filled
            # black_mask = (target_data == 0)  # area that should stay black
            #
            # LOSS = (0.6 * F.l1_loss(output_data[holes_mask], target_data[holes_mask]) +
            #         0.4 * F.l1_loss(output_data[black_mask], target_data[black_mask]))


            # Test 6
            # # holes_mask = ((target_data > 0) & (input_data == 0)).float()  # Convert to float for multiplication
            # # black_mask = (target_data == 0)  # area that should stay black
            # #
            # # diff = torch.abs(output_data - target_data)
            # # masked_diff = diff * (holes_mask + black_mask)
            # #
            # # # Normalize by the number of pixels in the mask
            # # LOSS = masked_diff.sum() / holes_mask.sum()
            # # LOSS += loss_functions.perceptual_loss(out=output_data, target=target_data, channels=1, device=self.args.device)
            # # LOSS += loss_functions.bce_dice_loss(out=output_data, target=target_data)
            #
            # # keep_mask1 = (input_data > 0).float()  # Area that should stay unchanged
            # # black_mask1 = (target_data == 0).float()  # Area that should stay black
            # # fill_mask1 = ((target_data > 0) & (input_data == 0)).float()   # Area that should be filled
            #
            # fill_mask1 = (torch.abs(target_data - input_data) > 0).float()  # Area that should be filled
            # black_mask1 = (target_data == 0).float()  # Area that should stay black
            # keep_mask1 = 1.0 - (fill_mask1 + black_mask1)  # Area that should stay unchanged
            #
            # # fill_weight = black_mask1.sum() / np.ones(shape=black_mask1.shape).sum() * 100
            # # black_weight = fill_mask1.sum() / np.ones(shape=keep_mask1.shape).sum() * 100
            # # keep_weight = 100 - (fill_weight + black_weight)
            # # weighted_mask1 = fill_weight * fill_mask1 + black_weight * black_mask1 + keep_weight * keep_mask1
            #
            # # weighted_mask1 = 0.80 * fill_mask1 + 0.15 * keep_mask1 + 0.05 * black_mask1
            #
            # weighted_mask1 = 80.0 * fill_mask1 + 15.0 * keep_mask1 + 5.0 * black_mask1
            #
            # # abs_diff1 = torch.abs(output_data - target_data)
            # # masked_abs_diff1 = abs_diff1 * fill_mask1
            # # diff2 = torch.abs(output_data) * black_mask1 + torch.abs(output_data) * keep_mask1
            #
            # output_data_1 = output_data * weighted_mask1
            # target_data_1 = target_data * weighted_mask1
            #
            # # Normalize by the number of pixels in the mask
            # LOSS = loss_functions.l1_loss(out=output_data_1, target=target_data_1, reduction='sum')
            #
            # # LOSS = masked_abs_diff1.sum()
            # # LOSS += loss_functions.bce_loss(out=output_data, target=target_data, reduction='sum')
            #
            # # LOSS += loss_functions.perceptual_loss(out=output_data, target=target_data, channels=1, device=self.args.device)
            # # LOSS += self.perceptual_loss(output=output_data, target=target_data)
            #
            # # LOSS = loss_functions.l1_loss(out=output_data, target=target_data, reduction='sum')
            #
            # # LOSS += 100 * loss_functions.dice_loss(out=output_data, target=target_data)


            # Test 7
            # fill_mask1 = (torch.abs(target_data - input_data) > 0).float()  # Area that should be filled
            # black_mask1 = (target_data == 0).float()  # Area that should stay black
            # keep_mask1 = 1.0 - (fill_mask1 + black_mask1)  # Area that should stay unchanged
            #
            # weighted_mask1 = 80.0 * fill_mask1 + 15.0 * keep_mask1 + 5.0 * black_mask1
            #
            # output_data_1 = output_data * weighted_mask1
            # target_data_1 = target_data * weighted_mask1
            #
            # # Normalize by the number of pixels in the mask
            # LOSS = loss_functions.l1_loss(out=output_data_1, target=target_data_1, reduction='sum')
            # # LOSS += self.perceptual_loss(output=output_data, target=target_data)


            # Test 8 - Parse2022 - 48
            # fill_mask1 = (torch.abs(target_data - input_data) > 0).float()  # Area that should be filled
            # black_mask1 = (target_data == 0).float()  # Area that should stay black
            # keep_mask1 = 1.0 - (fill_mask1 + black_mask1)  # Area that should stay unchanged
            #
            # weighted_mask1 = 8.0 * fill_mask1 + 1.5 * keep_mask1 + 0.5 * black_mask1
            # output_data_1 = output_data * weighted_mask1
            # target_data_1 = target_data * weighted_mask1
            #
            # LOSS = loss_functions.l1_loss(out=output_data_1, target=target_data_1, reduction='sum')
            # LOSS += 100 * loss_functions.vgg_loss.TVLoss(p=2).to(self.args.device)(output_data)


            # Test 9 - PCD
            # fill_mask1 = (torch.abs(target_data - input_data) > 0).float()  # Area that should be filled
            # black_mask1 = (target_data == 0).float()  # Area that should stay black
            # keep_mask1 = 1.0 - (fill_mask1 + black_mask1)  # Area that should stay unchanged
            #
            # weighted_mask1 = 10.0 * fill_mask1 + 1.0 * keep_mask1 + 0.05 * black_mask1
            # output_data_1 = output_data * weighted_mask1
            # target_data_1 = target_data * weighted_mask1
            #
            # LOSS = loss_functions.l1_loss(out=output_data_1, target=target_data_1, reduction='sum')

            # Test 10 - PCD
            fill_mask1 = (torch.abs(target_data - input_data) > 0).float()  # Area that should be filled
            black_mask1 = (target_data == 0).float()  # Area that should stay black
            keep_mask1 = 1.0 - (fill_mask1 + black_mask1)  # Area that should stay unchanged

            weighted_mask1 = 80.0 * fill_mask1 + 15.0 * keep_mask1 + 5.0 * black_mask1
            output_data_1 = output_data * weighted_mask1
            target_data_1 = target_data * weighted_mask1

            LOSS = loss_functions.l1_loss(out=output_data_1, target=target_data_1, reduction='sum')
            # LOSS = loss_functions.reconstruction_loss(out=output_data_1, target=target_data_1)
            # LOSS += 100 * loss_functions.vgg_loss.TVLoss(p=2).to(self.args.device)(output_data)


            # def weighted_mask_loss(predicted, target, mask, hole_weight=2.0):
            #     """
            #     Compute weighted binary cross-entropy loss focusing on holes.
            #     :param predicted: Predicted image (B, C, H, W)
            #     :param target: Ground truth image (B, C, H, W)
            #     :param mask: Binary mask (1 for holes, 0 for non-holes)
            #     :param hole_weight: Weight for the hole regions
            #     :return: Scalar loss
            #     """
            #     # Separate loss contributions
            #     hole_loss = F.binary_cross_entropy(predicted * mask, target * mask, reduction='sum')
            #     non_hole_loss = F.binary_cross_entropy(predicted * (1 - mask), target * (1 - mask), reduction='sum')
            #
            #     # Normalize by the number of pixels
            #     hole_loss /= mask.sum() + 1e-8  # Avoid division by zero
            #     non_hole_loss /= (1 - mask).sum() + 1e-8
            #
            #     # Weighted combination
            #     total_loss = hole_weight * hole_loss + non_hole_loss
            #     return total_loss
            #
            # def adaptive_weighted_loss(predicted, target, mask, base_weight=2.0):
            #     """
            #     Adaptive weighting for holes based on their size.
            #     :param predicted: Predicted image (B, C, H, W)
            #     :param target: Ground truth image (B, C, H, W)
            #     :param mask: Binary mask (1 for holes, 0 for non-holes)
            #     :param base_weight: Base weight for hole loss
            #     :return: Scalar loss
            #     """
            #     hole_area = mask.sum()
            #     adaptive_weight = base_weight * (1.0 + 1.0 / (hole_area + 1e-8))  # Larger weight for smaller holes
            #
            #     hole_loss = F.binary_cross_entropy(predicted * mask, target * mask, reduction='sum')
            #     non_hole_loss = F.binary_cross_entropy(predicted * (1 - mask), target * (1 - mask), reduction='sum')
            #
            #     hole_loss /= hole_area + 1e-8
            #     non_hole_loss /= (1 - mask).sum() + 1e-8
            #
            #     total_loss = adaptive_weight * hole_loss + non_hole_loss
            #     return total_loss
            #
            # def total_variation_loss(image):
            #     """
            #     Total variation loss for smoothness.
            #     :param image: Predicted image (B, C, H, W)
            #     :return: Scalar loss
            #     """
            #     loss = torch.sum(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :])) + \
            #            torch.sum(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]))
            #     return loss
            #
            # keep_mask1 = (input_data > 0).float()  # Area that should stay unchanged
            # black_mask1 = (target_data == 0).float()  # Area that should stay black
            # fill_mask1 = 1.0 - (keep_mask1 + black_mask1)  # Area that should be filled
            #
            # # weighted_mask1 = 0.80 * fill_mask1 + 0.15 * keep_mask1 + 0.05 * black_mask1
            #
            # # Normalize by the number of pixels in the mask
            # LOSS = adaptive_weighted_loss(predicted=output_data, target=target_data, mask=fill_mask1)
            # # LOSS += total_variation_loss(image=output_data)
            # LOSS += loss_functions.perceptual_loss(out=output_data, target=target_data, channels=1, device=self.args.device)
            # LOSS += loss_functions.bce_dice_loss(out=output_data, target=target_data)

            # LOSS += loss_functions.perceptual_loss(out=output_data_1, target=target_data_1, channels=1,device=self.args.device)
            # LOSS += loss_functions.bce_dice_loss(out=output_data_1, target=target_data_1)

            # Summary
            # TODO: Replace mask usage to multiple and check differentiation
            # TODO: Make sure crop include lines that needs to be connected (that the completion is not to a line out of image)
            # TODO: Output - input (in the network itself)
            # TODO: change the rotations of the pipes planes
            # TODO: remove use of PCDs


            # # Test 4
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


        if output_confidence_data is not None:
            # Confidence loss V1
            # target_confidence_data = (target_data != 0).float()
            # LOSS += F.binary_cross_entropy(output_confidence_data, target_confidence_data)


            # Confidence loss V2
            # target_confidence_data = (target_data != 0).float()
            # LOSS += F.binary_cross_entropy(output_confidence_data, target_confidence_data)


            # Confidence loss V3
            # target_holes_confidence_data = (target_data[holes_mask] > 0).float()
            # target_black_confidence_data = target_data[black_mask]
            #
            # LOSS += (0.2 * F.binary_cross_entropy(output_confidence_data[holes_mask], target_holes_confidence_data) +
            #          0.8 * F.binary_cross_entropy(output_confidence_data[black_mask], target_black_confidence_data))


            # Confidence loss V4
            # holes_mask = ((target_data - input_data) > 0)  # area that should be filled
            # black_mask = (target_data == 0)  # area that should stay black
            #
            # target_holes_confidence_data = (target_data[holes_mask] > 0).float()
            # target_black_confidence_data = target_data[black_mask]
            #
            # LOSS += (0.2 * F.binary_cross_entropy(output_confidence_data[holes_mask], target_holes_confidence_data) +
            #          0.8 * F.binary_cross_entropy(output_confidence_data[black_mask], target_black_confidence_data))


            # Confidence loss V5
            # holes_mask = ((target_data > 0) & (input_data == 0)).float()  # Convert to float for multiplication
            # black_mask = (target_data == 0)  # area that should stay black
            # target_confidence_data = (target_data > 0).float()
            #
            # diff = torch.abs(output_confidence_data - target_confidence_data)
            # masked_diff = diff * (holes_mask + black_mask)
            #
            # # Normalize by the number of pixels in the mask
            # LOSS += masked_diff.sum() / holes_mask.sum()
            # LOSS += loss_functions.perceptual_loss(out=output_confidence_data, target=target_confidence_data, channels=1, device=self.args.device)
            # LOSS += loss_functions.bce_dice_loss(out=output_confidence_data, target=target_confidence_data)

            # Confidence loss V6
            keep_mask = (input_data > 0).float()  # Area that should stay unchanged
            black_mask = (target_data == 0).float()  # Area that should stay black
            fill_mask = 1.0 - (keep_mask + black_mask)  # Area that should be filled

            weighted_mask = 85 * fill_mask + 10 * black_mask + 5 * keep_mask

            target_confidence_data = (target_data > 0).float()
            abs_diff = torch.abs(output_confidence_data - target_confidence_data)
            masked_abs_diff = abs_diff * weighted_mask

            # Normalize by the number of pixels in the mask
            LOSS += masked_abs_diff.sum()
            # LOSS += loss_functions.perceptual_loss(out=output_confidence_data, target=target_confidence_data, channels=1, device=self.args.device)
            # LOSS += loss_functions.bce_dice_loss(out=output_confidence_data, target=target_confidence_data)

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
                print(
                    '[Train] Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {}'.format(
                        epoch,
                        batch_idx * len(input_data),
                        len(self.train_loader.dataset),
                        100. * batch_idx / len(self.train_loader),
                        loss.item() / len(input_data)
                    )
                )

        train_avg_loss = train_loss / len(self.train_loader.dataset)
        print('> [Train] Epoch: {} Average loss: {}'.format(epoch, train_avg_loss))
        return train_avg_loss

    def _test(self, epoch):
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
        print('> [Test] Epoch: {}, Average Loss: {}'.format(epoch, test_avg_loss))
        return test_avg_loss

    def train(self, use_weights=False):
        if use_weights is True:
            print("Loading Model Weights")
            self.model.load_state_dict(torch.load(self.args.weights_filepath))

        print(f"[Model: '{self.model.model_name}'] Training...")
        try:
            for epoch in range(1, self.args.epochs + 1):
                train_avg_loss = self._train(epoch=epoch)
                test_avg_loss = self._test(epoch=epoch)
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
    def predict(self, max_batches_to_plot=2):
        print(f"[Model: '{self.model.model_name}'] Predicting...")
        os.makedirs(name=self.args.results_path, exist_ok=True)

        # Load model weights
        if os.path.exists(self.args.weights_filepath):
            self.model.load_state_dict(torch.load(self.args.weights_filepath))
        self.model.eval()

        iter_data = iter(self.test_loader)
        with torch.no_grad():
            batches_to_plot = min(len(self.test_loader), max_batches_to_plot)
            for batch_idx in range(batches_to_plot):
                print(f"Batch {batch_idx + 1}/{batches_to_plot}")

                # Get the images from the test loader
                batch_num = batch_idx + 1
                input_data, target_data = next(iter_data)

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
                output_data = self.model(input_data)

                if "confidence map" in getattr(self.model, "additional_tasks", list()):
                    output_data, output_confidence_data = output_data
                    map_merge_data = torch.where(output_confidence_data > 0.5, output_data, 0)
                else:
                    output_confidence_data = None
                    map_merge_data = output_data.clone()

                # TODO: Threshold
                # apply_threshold(output_images, 0.5)

                merge_data = torch.where(input_data > 0, input_data, map_merge_data)

                # Detach the images from the cuda and move them to CPU
                if self.args.cuda is True:
                    input_data = input_data.cpu()
                    target_data = target_data.cpu()
                    output_data = output_data.cpu()
                    merge_data = merge_data.cpu()

                    if output_confidence_data is not None:
                        output_confidence_data = output_confidence_data.cpu()
                        map_merge_data = map_merge_data.cpu()

                # Convert (b, 6, w, h) to (6*b, 1, w, h) - Trees2DV2
                if input_data.shape[1] == 6:
                    x, y = self.args.input_size[1:]
                    input_data = input_data.view(-1, 1, x, y)
                    target_data = target_data.view(-1, 1, x, y)
                    output_data = output_data.view(-1, 1, x, y)
                    fusion_data = fusion_data.view(-1, 1, x, y)

                    if output_confidence_data is not None:
                        output_confidence_data = output_confidence_data.view(-1, 1, x, y)
                        map_merge_data = map_merge_data.view(-1, 1, x, y)

                #################
                # Visualization #
                #################

                plotting_data_list = [
                    {"Title": "Input", "Data": input_data},
                    {"Title": "Target", "Data": target_data},
                    {"Title": "Output", "Data": output_data}
                ]

                # Handle additional_tasks
                if "confidence map" in getattr(self.model, "additional_tasks", list()):
                    additional_plotting_data_list = [
                        {"Title": "Confidence", "Data": output_confidence_data},
                        {"Title": "Map Merge", "Data": map_merge_data}
                    ]
                    plotting_data_list += additional_plotting_data_list

                plotting_data_list.append({"Title": "Merge", "Data": merge_data})

                apply_cleanup = True
                if apply_cleanup is True:
                    # TODO: Remove noise created by the model that doesn't connect components (add to predict pipeline if works)
                    pass

                # Create a grid of images
                columns = len(plotting_data_list)
                rows = input_data.size(0)
                fig = plt.figure(figsize=(columns + 0.5, rows + 0.5))
                ax = []
                for row_idx in tqdm(range(rows)):
                    for col_idx in range(columns):
                        data: torch.Tensor = plotting_data_list[col_idx]["Data"]
                        ax.append(fig.add_subplot(rows, columns, row_idx * columns + col_idx + 1))
                        numpy_image = data[row_idx].numpy()
                        plt.imshow(np.transpose(numpy_image, (1, 2, 0)), cmap='gray')

                for col_idx in range(columns):
                    title: str = plotting_data_list[col_idx]["Title"]
                    ax[col_idx].set_title(f"{title}:")

                fig.tight_layout()
                save_filename = os.path.join(self.args.results_path, f"{self.args.dataset}_{batch_num}.png")
                plt.savefig(save_filename)
                wandb.log(
                    data={f"Batch {batch_num} - Predict Plots": wandb.Image(plt)}
                )
