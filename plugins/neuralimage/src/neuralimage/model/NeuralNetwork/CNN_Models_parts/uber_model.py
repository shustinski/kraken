"""CNN_Models: uber model."""
from __future__ import annotations
from torch import nn
from neuralimage.model.NeuralNetwork.registrator import *
from neuralimage.model.NeuralNetwork.blocks import *

from neuralimage.model.NeuralNetwork.CNN_Models_parts.dense_residual_block import (
    DenseResidualBlock,
    DepthwiseSeparableConv,
    UpsampleBlock,
)

# @register_model(name='CNN Uber')
class UberModel(nn.Module):
    """ Fully Convolutional Network with Maximum Optimization """

    def __init__(self, input_channels=1):
        super().__init__()

        self.encoder = nn.Sequential(
            DepthwiseSeparableConv(input_channels, 64),
            DenseResidualBlock(64),
            DepthwiseSeparableConv(64, 128),
            DenseResidualBlock(128),
            DepthwiseSeparableConv(128, 256),
            SEBlock(256),
            DenseResidualBlock(256),
            nn.AdaptiveAvgPool2d(1)
        )

        self.decoder = nn.Sequential(
            UpsampleBlock(256, 128),  # 1 -> 2
            DenseResidualBlock(128),
            UpsampleBlock(128, 64),  # 2 -> 4
            DenseResidualBlock(64),
            UpsampleBlock(64, 32),  # 4 -> 8
            DenseResidualBlock(32),
            UpsampleBlock(32, 16),  # 8 -> 16
            DenseResidualBlock(16),
            UpsampleBlock(16, 8),  # 16 -> 32
            DenseResidualBlock(8),
            UpsampleBlock(8, 4),  # 32 -> 64
            DenseResidualBlock(4),
            UpsampleBlock(4, 2),  # 64 -> 128
            DenseResidualBlock(2),
            UpsampleBlock(2, 1),  # 128 -> 256
            DepthwiseSeparableConv(1, input_channels)  # СѓР¶Рµ 256x256
            # UpsampleBlock(256, 128),
            # DenseResidualBlock(128),
            # UpsampleBlock(128, 64),
            # DenseResidualBlock(64),
            # UpsampleBlock(64, 32),
            # DepthwiseSeparableConv(32, input_channels),
            # nn.Tanh()
        )

    def forward(self, x, tta=False):
        if tta:
            base = self.forward(x, tta=False)
            vertical = self.forward(x.flip(2), tta=False).flip(2)
            horizontal = self.forward(x.flip(3), tta=False).flip(3)
            return (base + vertical + horizontal) / 3

        x = self.encoder(x)
        x = self.decoder(x)

        return x
