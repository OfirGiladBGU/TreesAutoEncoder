from torch import nn


# Model
class Network(nn.Module):
    def __init__(self, args):
        super(Network, self).__init__()
        self.model_name = 'gap_cnn'
        self.input_size = args.input_size
        self.c = self.input_size[0]

        self.layer1 = nn.Conv2d(in_channels=self.c, out_channels=24, kernel_size=5, stride=2, padding=2)  # scale: 1/2
        self.layer2 = nn.Conv2d(in_channels=24, out_channels=64, kernel_size=3, stride=2, padding=1)  # scale: 1/4
        self.layer3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)  # scale: 1/4
        self.layer4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)  # scale: 1/8
        self.layer5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)  # scale: 1/8
        self.layer6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1)  # scale: 1/16
        self.layer7 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)  # scale: 1/16

        self.layer8U = nn.Upsample(scale_factor=2, mode='nearest')  # scale: 1/8
        self.layer8 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)  # scale: 1/8
        self.layer9 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)  # scale: 1/8

        self.layer10U = nn.Upsample(scale_factor=2, mode='nearest')  # scale: 1/4
        self.layer10 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)  # scale: 1/4
        self.layer11 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)  # scale: 1/4

        self.layer12U = nn.Upsample(scale_factor=2, mode='nearest')  # scale: 1/2
        self.layer12 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1)  # scale: 1/2
        self.layer13 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1)  # scale: 1/2

        self.layer14U = nn.Upsample(scale_factor=2, mode='nearest')  # scale: 1
        self.layer14 = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3, stride=1, padding=1)  # scale: 1
        self.layer15 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1)  # scale: 1
        self.layer16 = nn.Conv2d(in_channels=1, out_channels=self.c, kernel_size=3, stride=1, padding=1)  # scale: 1

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.relu(self.layer4(x))
        x = self.relu(self.layer5(x))
        x = self.relu(self.layer6(x))
        x = self.relu(self.layer7(x))
        x = self.relu(self.layer8U(x))
        x = self.relu(self.layer8(x))
        x = self.relu(self.layer9(x))
        x = self.relu(self.layer10U(x))
        x = self.relu(self.layer10(x))
        x = self.relu(self.layer11(x))
        x = self.relu(self.layer12U(x))
        x = self.relu(self.layer12(x))
        x = self.relu(self.layer13(x))
        x = self.relu(self.layer14U(x))
        x = self.relu(self.layer14(x))
        x = self.relu(self.layer15(x))
        x = self.sigmoid(self.layer16(x))
        return x
