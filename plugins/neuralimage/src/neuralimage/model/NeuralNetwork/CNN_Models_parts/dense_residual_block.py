"""CNN_Models: dense residual block."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from neuralimage.model.NeuralNetwork.registrator import *
from neuralimage.model.NeuralNetwork.blocks import *

class DepthwiseSeparableConv(nn.Module):
    """ EfficientNet-Style Depthwise Separable Convolution """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels,
                                   bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class DenseResidualBlock(nn.Module):
    """ Dense Residual Block with Multi-Scale Feature Fusion """

    def __init__(self, in_channels, growth_rate=32):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, growth_rate)
        self.conv2 = DepthwiseSeparableConv(in_channels + growth_rate, growth_rate)
        self.conv3 = DepthwiseSeparableConv(in_channels + 2 * growth_rate, in_channels)

    def forward(self, x):
        out1 = F.gelu(self.conv1(x))
        out2 = F.gelu(self.conv2(torch.cat([x, out1], dim=1)))
        out3 = self.conv3(torch.cat([x, out1, out2], dim=1))
        return x + out3  # Residual connection

class UpsampleBlock(nn.Module):
    """ Learnable Deconvolution for High-Quality Upsampling """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return F.gelu(self.conv(x))
