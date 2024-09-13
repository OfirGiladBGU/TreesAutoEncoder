import argparse
import torch.nn as nn


class Network2D(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super(Network2D, self).__init__()

        self.model_name = 'ae_6_2d_to_6_2d'
        self.input_size = args.input_size

        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, stride=2, padding=1),  # (batch_size, 64, H/2, W/2)
            nn.ReLU(True)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (batch_size, 128, H/4, W/4)
            nn.ReLU(True)
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # (batch_size, 256, H/8, W/8)
            nn.ReLU(True)
        )

        # Decoder
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            # (batch_size, 128, H/4, W/4)
            nn.ReLU(True)
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            # (batch_size, 64, H/2, W/2)
            nn.ReLU(True)
        )
        self.decoder3 = nn.Sequential(
            nn.ConvTranspose2d(64, 6, kernel_size=3, stride=2, padding=1, output_padding=1),  # (batch_size, 6, H, W)
            nn.Sigmoid()  # To normalize the output to [0, 1]
        )

    def forward(self, x):
        z = x

        # Encoding
        z1 = self.encoder1(z)
        z2 = self.encoder2(z1)
        z3 = self.encoder3(z2)

        # Decoding with skip connections
        z = self.decoder1(z3)
        z = self.decoder2(z + z2)  # Skip connection
        z = self.decoder3(z + z1)  # Skip connection

        return z
