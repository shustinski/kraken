"""CNN_Models: medium optimised."""
from __future__ import annotations
from torch import nn
from neuralimage.model.NeuralNetwork.registrator import *
from neuralimage.model.NeuralNetwork.blocks import *

# @register_model(name='MO 1.3M')
class MediumOptimised(nn.Module):
    def __init__(self, input_channels = 1):
        super(MediumOptimised, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, stride=2, padding=0),

            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, stride=2, padding=0),

            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, stride=2, padding=0),

            nn.Conv2d(128, 160, kernel_size=3, padding=1),
            nn.BatchNorm2d(160),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, stride=2, padding=0),

            nn.Conv2d(160, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, stride=2, padding=0),
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(192, 160, kernel_size=3, padding=1),
            nn.BatchNorm2d(160),
            nn.LeakyReLU(0.2),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(160, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.2),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
