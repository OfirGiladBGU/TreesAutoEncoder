import torch
import torch.nn as nn


class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layer1 = ConvBlock2D(1, 16)
        self.layer2 = ConvBlock2D(16, 32)
        self.layer3 = ConvBlock2D(32, 64)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        return x1, x2, x3


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.layer1 = nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.layer2 = nn.ConvTranspose3d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.layer3 = nn.ConvTranspose3d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.sigmoid(x)


class MultiView3DReconstruction(nn.Module):
    def __init__(self, args=None):
        super(MultiView3DReconstruction, self).__init__()

        self.model_name = 'multi_view_3d_reconstruction'
        if args is not None:
            self.input_size = args.input_size

        self.encoders = nn.ModuleList([Encoder() for _ in range(6)])
        self.fc = nn.Linear(64 * 32 * 32 * 6, 64 * 4 * 4 * 4)
        self.decoder = Decoder()

    def forward(self, views):
        # views shape: (batch_size, 6, 1, 32, 32)
        batch_size = views.size(0)
        encoded_views = []
        for i in range(6):
            view = views[:, i, :, :, :]  # Extract the i-th view from each batch
            encoded_view = self.encoders[i](view)[-1]  # Only use the last feature map
            encoded_views.append(encoded_view)

        encoded_views = torch.cat(encoded_views, dim=1)  # Concatenate along the channel dimension
        encoded_views = encoded_views.view(batch_size, -1)
        x = self.fc(encoded_views)
        x = x.view(batch_size, 64, 4, 4, 4)
        x = self.decoder(x)
        return x


# Example usage
if __name__ == '__main__':
    model = MultiView3DReconstruction().cuda()
    views = torch.randn(2, 6, 1, 32, 32).cuda()  # Dummy input for batch_size=2 and 6 views per sample
    output = model(views)
    print(output.shape)  # Should be [2, 1, 32, 32, 32]
