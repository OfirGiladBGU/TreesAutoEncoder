import numpy as np
import torch
import torch.nn as nn

import pytorch_ssim
from trainer import vgg_loss
from torchmetrics.segmentation import MeanIoU
from torcheval.metrics.functional import multiclass_f1_score


#####################
# Utility functions #
#####################
def reshape_inputs(input_data, input_size):
    input_data = input_data.view(-1, *input_size)
    return input_data


#########################
# Common Loss functions #
#########################
def bce_loss(out, target, reduction='sum'):
    loss_fn = nn.BCELoss(reduction=reduction)
    return loss_fn(out, target)


def mse_loss(out, target, reduction='sum'):
    loss_fn = nn.MSELoss(reduction=reduction)
    return loss_fn(out, target)


def l1_loss(out, target, reduction='sum'):
    loss_fn = nn.L1Loss(reduction=reduction)
    return loss_fn(out, target)


###################
# GPT Suggestions #
###################
def reconstruction_loss(out, target, reduction='sum'):
    return mse_loss(out, target, reduction=reduction) / (out.size(0) * out.size(1))


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
    reshaped_out = reshape_inputs(input_data=out, input_size=(input_size[1] * input_size[2]))
    reshaped_target = reshape_inputs(input_data=target, input_size=(input_size[1] * input_size[2]))

    scaled_out = reshaped_out * 255
    scaled_out = scaled_out.int()

    scaled_target = reshaped_target * 255
    scaled_target = scaled_target.int()

    y_pred = scaled_out
    y_true = scaled_target
    return torch.mean(torch.square(torch.cumsum(y_true, dim=-1) - torch.cumsum(y_pred, dim=-1)), dim=-1)


def unfilled_holes_loss(out, target, original):
    target_filled_holes = target - original
    target_filled_holes[target_filled_holes < 0] = 0

    out_missing_holes = target_filled_holes - out
    out_missing_holes[out_missing_holes < 0] = 0

    diff = torch.sum(out_missing_holes) # / out.size(0)
    return diff


# TODO: fix
def weighted_pixels_diff_loss(out, target, original):
    output_diff = out - original

    # Make sure to keep the non-zero pixels stay with the same color
    misclassified_pixels = output_diff.clone()
    misclassified_pixels = torch.where(target != 0, misclassified_pixels, torch.tensor(0.0))
    misclassified_pixels_error = torch.sum(torch.abs(misclassified_pixels)) # / out.size(0)

    # Make sure to prevent black pixels to change to gray/white
    new_pixels_error = output_diff.clone()
    new_pixels_error = torch.where(target == 0, new_pixels_error, torch.tensor(0.0))
    new_pixels_error = torch.sum(torch.abs(new_pixels_error)) # / out.size(0)

    return 0.8 * misclassified_pixels_error + 0.2 * new_pixels_error


# Dice Loss and BCE Dice Loss
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice


# Weighted BCE Loss
class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=1.0):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        # Calculate the weights: higher for positive (1) and lower for negative (0)
        weights = torch.where(targets == 1, self.pos_weight, 1.0)
        # Apply BCE with weights
        loss = nn.functional.binary_cross_entropy(inputs, targets, weight=weights)
        return loss


class BCEDiceLoss(nn.Module):
    def __init__(self, weight_bce=0.5, weight_dice=0.5):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs[0], targets[0])
        dice_loss = self.dice(inputs, targets)
        return self.weight_bce * bce_loss + self.weight_dice * dice_loss


# BCE + Dice Loss with weighted BCE
class WeightedBCEDiceLoss(nn.Module):
    def __init__(self, weight_bce=0.5, weight_dice=0.5, pos_weight=2.0):
        super(WeightedBCEDiceLoss, self).__init__()
        self.bce = WeightedBCELoss(pos_weight=pos_weight)
        self.dice = DiceLoss()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice

    def forward(self, inputs, targets):
        bce_loss_res = self.bce(inputs, targets)
        dice_loss_res = self.dice(inputs, targets)
        return self.weight_bce * bce_loss_res + self.weight_dice * dice_loss_res


def dice_loss(out, target):
    loss_fn = DiceLoss()
    return loss_fn(out, target)


def bce_dice_loss(out, target):
    loss_fn = BCEDiceLoss()
    return loss_fn(out, target)


def weighted_bce_dice_loss(out, target):
    loss_fn = WeightedBCEDiceLoss()
    return loss_fn(out, target)
