import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from trainer import vgg_loss


#####################
# Utility functions #
#####################
def reshape_inputs(input_data, input_size):
    input_data = input_data.view(-1, *input_size)
    return input_data


def downsample(input_data, scale=2):
    return F.interpolate(input_data, scale_factor=1 / scale, mode='bilinear', align_corners=False)


#############################
# Our custom loss functions #
#############################
def weighted_mask_loss(output, target, input, lambda_value=100.0, reduction='sum'):
    fill_mask = (torch.abs(target - input) > 0).float()  # Area that should be filled
    black_mask = (target == 0).float()  # Area that should stay black
    keep_mask = 1.0 - (fill_mask + black_mask)  # Area that should stay unchanged

    weighted_mask = lambda_value * (0.80 * fill_mask + 0.15 * keep_mask + 0.05 * black_mask)
    weighted_output = output * weighted_mask
    weighted_target = target * weighted_mask

    return l1_loss(output=weighted_output, target=weighted_target, reduction=reduction)


#########################
# Common Loss functions #
#########################
def bce_loss(output, target, reduction='sum'):
    loss_fn = nn.BCELoss(reduction=reduction)
    return loss_fn(output, target)


def mse_loss(output, target, reduction='sum'):
    loss_fn = nn.MSELoss(reduction=reduction)
    return loss_fn(output, target)


def l1_loss(output, target, reduction='sum'):
    loss_fn = nn.L1Loss(reduction=reduction)
    return loss_fn(output, target)


def reconstruction_loss(output, target, reduction='sum'):
    return mse_loss(output, target, reduction=reduction) / (output.size(0) * output.size(1))


# From VGG Loss
def perceptual_loss(output, target, channels, device):
    if channels == 1:
        # Change shapes to 3 channels
        rgb_output = output.repeat(1, 3, 1, 1)
        rgb_target = target.repeat(1, 3, 1, 1)
    elif channels == 3:
        rgb_output = output
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

    loss = crit(rgb_output, rgb_target)
    return loss


def perceptual_loss_v2(output, target, device):
    import torchvision.models as models
    # vgg = models.vgg19(pretrained=True).features[:16].eval()  # Use the first few layers of VGG19 (Legacy)
    vgg = models.vgg19().features[:16].eval()  # Use the first few layers of VGG19
    vgg = vgg.to(device)

    output_3ch = output.repeat(1, 3, 1, 1)
    target_3ch = target.repeat(1, 3, 1, 1)

    output_features = vgg(output_3ch)
    target_features = vgg(target_3ch)
    return F.l1_loss(output_features, target_features)


def edge_loss(output, target, device):
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

    g1_x = convx(output)
    g2_x = convx(target)
    g1_y = convy(output)
    g2_y = convy(target)

    # Option 2
    # g1_x = nn.functional.conv2d(input=output, weight=weights_x, stride=1, padding=1)
    # g2_x = nn.functional.conv2d(input=target, weight=weights_x, stride=1, padding=1)
    # g1_y = nn.functional.conv2d(input=output, weight=weights_y, stride=1, padding=1)
    # g2_y = nn.functional.conv2d(input=target, weight=weights_y, stride=1, padding=1)

    # Calculate the gradient magnitude
    g_1 = torch.sqrt(torch.pow(g1_x, 2) + torch.pow(g1_y, 2))
    g_2 = torch.sqrt(torch.pow(g2_x, 2) + torch.pow(g2_y, 2))

    return torch.mean((g_1 - g_2).pow(2))


##############################
# Less common loss functions #
##############################
def mean_iou(output, target):
    from torchmetrics.segmentation import MeanIoU

    scaled_output = output * 255
    scaled_output = scaled_output.int()

    scaled_target = target * 255
    scaled_target = scaled_target.int()

    miou = MeanIoU(num_classes=255)
    return miou(scaled_output, scaled_target)


def total_variation_loss(output, target, p, device):
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
    return tv_loss(output, target)


def total_variation_loss_v2(output):
    return (
        torch.sum(torch.abs(output[:, :, :-1, :] - output[:, :, 1:, :])) +
        torch.sum(torch.abs(output[:, :, :, :-1] - output[:, :, :, 1:]))
    )


def ssim_loss(output, target):
    import pytorch_ssim
    return pytorch_ssim.ssim(output, target)


def f1_loss(output, target):
    from torcheval.metrics.functional import multiclass_f1_score

    scaled_output = output * 255
    scaled_output = scaled_output.int()

    scaled_target = target * 255
    scaled_target = scaled_target.int()

    return multiclass_f1_score(scaled_output, scaled_target, num_classes=255)


def earth_mover_distance(output, target, input_size=(1, 64, 64)):
    reshaped_output = reshape_inputs(input_data=output, input_size=(input_size[1] * input_size[2]))
    reshaped_target = reshape_inputs(input_data=target, input_size=(input_size[1] * input_size[2]))

    scaled_output = reshaped_output * 255
    scaled_output = scaled_output.int()

    scaled_target = reshaped_target * 255
    scaled_target = scaled_target.int()

    y_pred = scaled_output
    y_true = scaled_target
    return torch.mean(torch.square(torch.cumsum(y_true, dim=-1) - torch.cumsum(y_pred, dim=-1)), dim=-1)


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


def dice_loss(output, target):
    loss_fn = DiceLoss()
    return loss_fn(output, target)


def bce_dice_loss(output, target):
    loss_fn = BCEDiceLoss()
    return loss_fn(output, target)


def weighted_bce_dice_loss(output, target):
    loss_fn = WeightedBCEDiceLoss()
    return loss_fn(output, target)
