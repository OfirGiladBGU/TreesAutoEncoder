import numpy as np
import torch
import torch.nn as nn

import pytorch_ssim


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


def ssim_loss(out, target, input_size=(1, 64, 64)):
    return pytorch_ssim.ssim(
        out.view(-1, input_size[0], input_size[1], input_size[2]),
        target.view(-1, input_size[0], input_size[1], input_size[2])
    )
