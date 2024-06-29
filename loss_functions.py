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
def reshape_data(data, input_size):
    return data.view(-1, input_size[0], input_size[1], input_size[2])


def reshape_inputs(input_images, target_images, input_size):
    input_images = reshape_data(input_images, input_size)
    target_images = reshape_data(target_images, input_size)
    return input_images, target_images


###################
# GPT Suggestions #
###################
def reconstruction_loss(out, target, input_size=(1, 64, 64)):
    reshaped_out, reshaped_target = reshape_inputs(out, target, input_size)

    mse_loss = torch.nn.MSELoss()
    return mse_loss(reshaped_out, reshaped_target)


# From VGG Loss
def perceptual_loss(out, target, device, input_size=(1, 64, 64)):
    reshaped_out, reshaped_target = reshape_inputs(out, target, input_size)

    # Change shapes to 3 channels
    rgb_out = reshaped_out.repeat(1, 3, 1, 1)
    rgb_target = reshaped_target.repeat(1, 3, 1, 1)

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


def edge_loss(out, target, device, input_size=(1, 64, 64)):
    reshaped_out, reshaped_target = reshape_inputs(out, target, input_size)

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

    g1_x = convx(reshaped_out)
    g2_x = convx(reshaped_target)
    g1_y = convy(reshaped_out)
    g2_y = convy(reshaped_target)

    # Option 2
    # g1_x = nn.functional.conv2d(input=reshaped_out, weight=weights_x, stride=1, padding=1)
    # g2_x = nn.functional.conv2d(input=reshaped_target, weight=weights_x, stride=1, padding=1)
    # g1_y = nn.functional.conv2d(input=reshaped_out, weight=weights_y, stride=1, padding=1)
    # g2_y = nn.functional.conv2d(input=reshaped_target, weight=weights_y, stride=1, padding=1)

    # Calculate the gradient magnitude
    g_1 = torch.sqrt(torch.pow(g1_x, 2) + torch.pow(g1_y, 2))
    g_2 = torch.sqrt(torch.pow(g2_x, 2) + torch.pow(g2_y, 2))

    return torch.mean((g_1 - g_2).pow(2))


###############################
# Other useful loss functions #
###############################
def mean_iou(out, target, input_size=(1, 64, 64)):
    reshaped_out, reshaped_target = reshape_inputs(out, target, input_size)

    scaled_out = reshaped_out * 255
    scaled_out = scaled_out.int()

    scaled_target = reshaped_target * 255
    scaled_target = scaled_target.int()

    miou = MeanIoU(num_classes=255)
    return miou(scaled_out, scaled_target)


def total_variation_lost(out, target, device, input_size=(1, 64, 64)):
    reshaped_out, reshaped_target = reshape_inputs(out, target, input_size)

    tv_loss = vgg_loss.TVLoss(p=2).to(device)
    return tv_loss(reshaped_out, reshaped_target)


def ssim_loss(out, target, input_size=(1, 64, 64)):
    reshaped_out, reshaped_target = reshape_inputs(out, target, input_size)

    return pytorch_ssim.ssim(reshaped_out, reshaped_target)


def f1_loss(out, target, input_size=(1, 64, 64)):
    reshaped_out, reshaped_target = reshape_inputs(out, target, input_size)

    scaled_out = reshaped_out * 255
    scaled_out = scaled_out.int()

    scaled_target = reshaped_target * 255
    scaled_target = scaled_target.int()

    return multiclass_f1_score(scaled_out, scaled_target, num_classes=255)
