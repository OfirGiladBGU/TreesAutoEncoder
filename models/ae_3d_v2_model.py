import argparse
import torch.nn as nn


class Network3D(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super(Network3D, self).__init__()

        self.model_name = 'ae_3d_v2'
        self.input_size = args.input_size

        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3, stride=2, padding=1),  # (batch_size, 64, D/2, H/2, W/2)
            nn.ReLU(True)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),  # (batch_size, 128, D/4, H/4, W/4)
            nn.ReLU(True)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),  # (batch_size, 256, D/8, H/8, W/8)
            nn.ReLU(True)
        )

        # Decoder
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            # (batch_size, 128, D/4, H/4, W/4)
            nn.ReLU(True)
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            # (batch_size, 64, D/2, H/2, W/2)
            nn.ReLU(True)
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose3d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch_size, 1, D, H, W)
            nn.Sigmoid()  # To normalize the output to [0, 1]
        )

    def forward(self, x):
        # Encoding
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)

        # Decoding with skip connections
        x = self.decoder1(x3)
        x = self.decoder2(x + x2)  # Skip connection
        x = self.decoder3(x + x1)  # Skip connection

        return x
