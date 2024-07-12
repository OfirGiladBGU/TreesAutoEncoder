import numpy as np
import torch
import torch.nn as nn

import pytorch_ssim
import vgg_loss
from torchmetrics.segmentation import MeanIoU
from torcheval.metrics.functional import multiclass_f1_score


#####################
# Utility functions #
#####################
def reshape_inputs(input_images, target_images, input_size):
    input_images = input_images.view(-1, *input_size)
    target_images = target_images.view(-1, *input_size)
    return input_images, target_images


###################
# GPT Suggestions #
###################
def reconstruction_loss(out, target):
    mse_loss = torch.nn.MSELoss(reduction='sum')
    return mse_loss(out, target) / (out.size(0) * out.size(1))


# From VGG Loss
def perceptual_loss(out, target, channels, device):
    if channels == 1:
        # Change shapes to 3 channels
        rgb_out = out.repeat(1, 3, 1, 1)
        rgb_target = target.repeat(1, 3, 1, 1)
    elif channels == 3:
        rgb_out = out
        rgb_target = target
    else:
        raise ValueError("Channels must be 1 or 3")

    crit = vgg_loss.WeightedLoss(
        losses=[
            vgg_loss.VGGLoss(shift=2),
            nn.MSELoss(),
            vgg_loss.TVLoss(p=1)
        ],
        weights=[1, 40, 10]
    ).to(device)

    loss = crit(rgb_out, rgb_target)
    return loss


def edge_loss(out, target, device):
    # Sobel filters
    x_filter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    y_filter = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    weights_x = torch.from_numpy(x_filter).float().unsqueeze(0).unsqueeze(0)
    weights_y = torch.from_numpy(y_filter).float().unsqueeze(0).unsqueeze(0)

    weights_x = weights_x.to(device)
    weights_y = weights_y.to(device)

    # Option 1
    convx = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    convy = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    convx.weight = nn.Parameter(weights_x)
    convy.weight = nn.Parameter(weights_y)

    g1_x = convx(out)
    g2_x = convx(target)
    g1_y = convy(out)
    g2_y = convy(target)

    # Option 2
    # g1_x = nn.functional.conv2d(input=out, weight=weights_x, stride=1, padding=1)
    # g2_x = nn.functional.conv2d(input=target, weight=weights_x, stride=1, padding=1)
    # g1_y = nn.functional.conv2d(input=out, weight=weights_y, stride=1, padding=1)
    # g2_y = nn.functional.conv2d(input=target, weight=weights_y, stride=1, padding=1)

    # Calculate the gradient magnitude
    g_1 = torch.sqrt(torch.pow(g1_x, 2) + torch.pow(g1_y, 2))
    g_2 = torch.sqrt(torch.pow(g2_x, 2) + torch.pow(g2_y, 2))

    return torch.mean((g_1 - g_2).pow(2))


###############################
# Other useful loss functions #
###############################
def mean_iou(out, target):
    scaled_out = out * 255
    scaled_out = scaled_out.int()

    scaled_target = target * 255
    scaled_target = scaled_target.int()

    miou = MeanIoU(num_classes=255)
    return miou(scaled_out, scaled_target)


def total_variation_lost(out, target, p, device):
    """
    ``p=1`` yields the vectorial total variation norm. It is a generalization
    of the originally proposed (isotropic) 2D total variation norm (see
    (see https://en.wikipedia.org/wiki/Total_variation_denoising) for color
    images. On images with a single channel it is equal to the 2D TV norm.

    ``p=2`` yields a variant that is often used for smoothing out noise in
    reconstructions of images from neural network feature maps (see Mahendran
    and Vevaldi, "Understanding Deep Image Representations by Inverting
    Them", https://arxiv.org/abs/1412.0035)
    """
    tv_loss = vgg_loss.TVLoss(p=p).to(device)
    return tv_loss(out, target)


def ssim_loss(out, target):
    return pytorch_ssim.ssim(out, target)


def f1_loss(out, target):
    scaled_out = out * 255
    scaled_out = scaled_out.int()

    scaled_target = target * 255
    scaled_target = scaled_target.int()

    return multiclass_f1_score(scaled_out, scaled_target, num_classes=255)


def earth_mover_distance(out, target, input_size=(1, 64, 64)):
    reshaped_out, reshaped_target = reshape_inputs(out, target, input_size=(input_size[1] * input_size[2]))

    scaled_out = reshaped_out * 255
    scaled_out = scaled_out.int()

    scaled_target = reshaped_target * 255
    scaled_target = scaled_target.int()

    y_pred = scaled_out
    y_true = scaled_target
    return torch.mean(torch.square(torch.cumsum(y_true, dim=-1) - torch.cumsum(y_pred, dim=-1)), dim=-1)


def unfilled_holes_loss(out, target, original):
    target_holes = target - original
    target_holes[target_holes < 0] = 0

    out_holes = out - original
    out_holes[out_holes < 0] = 0

    unfilled_holes = target_holes - out_holes
    unfilled_holes[unfilled_holes < 0] = 0
    diff = torch.sum(unfilled_holes)
    return diff
