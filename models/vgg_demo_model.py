from torch import nn


# Model
class Network(nn.Module):
    def __init__(self, args):
        super(Network, self).__init__()
        self.model_name = 'vgg_ae_demo'
        self.input_size = args.input_size

        # Original
        if self.input_size == (3, 32, 32):
            encoder = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
                nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
                nn.AvgPool2d(2, ceil_mode=True),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                nn.AvgPool2d(2, ceil_mode=True),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                nn.AvgPool2d(2, ceil_mode=True),
                nn.Flatten(),
                nn.Linear(1024, 128), nn.Tanh(),
            )
            decoder = nn.Sequential(
                nn.Linear(128, 1024), nn.ReLU(),
                nn.Unflatten(-1, (64, 4, 4)),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
                nn.Conv2d(16, 3, 3, padding=1), nn.Sigmoid(),
            )
        # Modified
        elif self.input_size == (1, 64, 64):
            encoder = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
                nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
                nn.AvgPool2d(2, ceil_mode=True),  # 1x64x64 -> 16x32x32
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                nn.AvgPool2d(2, ceil_mode=True),  # 32x32x32 -> 32x16x16
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                nn.AvgPool2d(2, ceil_mode=True),  # 64x16x16 -> 64x8x8
                nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
                nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
                nn.AvgPool2d(2, ceil_mode=True),  # 128x8x8 -> 128x4x4
                nn.Flatten(),  # 128x4x4 -> 2048
                nn.Linear(2048, 256), nn.Tanh(),  # 2048 -> 256
            )
            decoder = nn.Sequential(
                nn.Linear(256, 2048), nn.ReLU(),  # 256 -> 2048
                nn.Unflatten(-1, (128, 4, 4)),  # 2048 -> 128x4x4
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 128x4x4 -> 128x8x8
                nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
                nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 64x8x8 -> 64x16x16
                nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 32x16x16 -> 32x32x32
                nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 16x32x32 -> 16x64x64
                nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
                nn.Conv2d(16, 1, 3, padding=1), nn.Sigmoid(),  # 16x64x64 -> 1x64x64
            )
        else:
            raise ValueError(f'Invalid input size: {self.input_size}')

        self.model = nn.Sequential(encoder, decoder)

    def forward(self, x):
        return self.model(x)
