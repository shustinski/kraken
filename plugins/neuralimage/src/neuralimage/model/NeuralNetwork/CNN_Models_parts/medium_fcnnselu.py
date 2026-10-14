"""CNN_Models: medium fcnnselu."""
from __future__ import annotations
from torch import nn
from neuralimage.model.NeuralNetwork.registrator import *
from neuralimage.model.NeuralNetwork.blocks import *

# @register_model(name='MS 720k')
class MediumFCNNSELU(nn.Module):
    def __init__(self, input_channels = 1):
        super(MediumFCNNSELU, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.SELU(),
            nn.MaxPool2d(2, stride=2, padding=0),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.SELU(),
            nn.MaxPool2d(2, stride=2, padding=0),

            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.SELU(),
            nn.MaxPool2d(2, stride=2, padding=0),

            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.SELU(),
            nn.MaxPool2d(2, stride=2, padding=0),

            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.SELU(),
            nn.MaxPool2d(2, stride=2, padding=0),
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.SELU(),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 96, kernel_size=3, padding=1),
            nn.SELU(),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.SELU(),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.SELU(),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.SELU(),

            nn.Conv2d(64, 1, kernel_size=3, padding=1),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
