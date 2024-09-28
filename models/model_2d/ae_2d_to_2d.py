# import argparse
# import torch.nn as nn
#
#
# # TODO: change kernel size
# # TODO: attention check
# class Network2D(nn.Module):
#     def __init__(self, args: argparse.Namespace):
#         super(Network2D, self).__init__()
#
#         self.model_name = 'ae_2d_to_2d'
#         self.input_size = args.input_size
#
#         # Encoder
#         self.encoder1 = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),  # (batch_size, 64, H/2, W/2)
#             nn.ReLU(True)
#         )
#         self.encoder2 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (batch_size, 128, H/4, W/4)
#             nn.ReLU(True)
#         )
#         self.encoder3 = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # (batch_size, 256, H/8, W/8)
#             nn.ReLU(True)
#         )
#
#         # Decoder
#         self.decoder1 = nn.Sequential(
#             nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
#             # (batch_size, 128, H/4, W/4)
#             nn.ReLU(True)
#         )
#         self.decoder2 = nn.Sequential(
#             nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
#             # (batch_size, 64, H/2, W/2)
#             nn.ReLU(True)
#         )
#         self.decoder3 = nn.Sequential(
#             nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch_size, 1, H, W)
#             nn.Sigmoid()  # To normalize the output to [0, 1]
#         )
#
#     def forward(self, x):
#         # Encoding
#         x1 = self.encoder1(x)
#         x2 = self.encoder2(x1)
#         x3 = self.encoder3(x2)
#
#         # Decoding with skip connections
#         x = self.decoder1(x3)
#         x = self.decoder2(x + x2)  # Skip connection
#         x = self.decoder3(x + x1)  # Skip connection
#
#         return x

import argparse
import torch
import torch.nn as nn

# Self-Attention Block
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        query = self.query(x).view(batch_size, -1, width * height)  # B x C/8 x N
        key = self.key(x).view(batch_size, -1, width * height)      # B x C/8 x N
        value = self.value(x).view(batch_size, -1, width * height)  # B x C x N

        attention = torch.bmm(query.permute(0, 2, 1), key)  # B x N x N
        attention = torch.softmax(attention, dim=-1)

        out = torch.bmm(value, attention.permute(0, 2, 1))  # B x C x N
        out = out.view(batch_size, C, width, height)

        out = self.gamma * out + x
        return out

class Network2D(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super(Network2D, self).__init__()

        self.model_name = 'ae_2d_to_2d'
        self.input_size = args.input_size

        # Encoder with larger kernels
        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2),  # (batch_size, 64, H/2, W/2)
            nn.ReLU(True)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),  # (batch_size, 128, H/4, W/4)
            nn.ReLU(True)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),  # (batch_size, 256, H/8, W/8)
            nn.ReLU(True)
        )

        # Self-Attention after the encoder
        self.attention = SelfAttention(256)

        # Decoder
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(True)
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(True)
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid()  # Normalize output to [0, 1]
        )

    def forward(self, x):
        # Encoding
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)

        # Attention block
        x3 = self.attention(x3)

        # Decoding with skip connections
        x = self.decoder1(x3)
        x = self.decoder2(x + x2)  # Skip connection
        x = self.decoder3(x + x1)  # Skip connection

        return x
