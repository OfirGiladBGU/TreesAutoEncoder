from torch import nn


# Model
class Network(nn.Module):
    def __init__(self, args):
        super(Network, self).__init__()

        if args.dataset == 'TreesV1':
            self.input_size = (1, 64, 64)
        else:
            self.input_size = (1, 32, 32)

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

        self.model = nn.Sequential(encoder, decoder)

    def forward(self, x):
        return self.model(x)
