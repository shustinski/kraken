"""CNN_Models: u net plus plus."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from neuralimage.model.NeuralNetwork.registrator import *
from neuralimage.model.NeuralNetwork.blocks import *

def _conv_bn_lrelu(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2),
    )

class _FCNNEncoderStage(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = _conv_bn_lrelu(in_channels, out_channels)

    def forward(self, x):
        return self.block(x)

class _FCNNDecoderStage(nn.Module):
    def __init__(self, in_channels, out_channels, *, blocks_count=1):
        super().__init__()
        layers = [nn.Upsample(scale_factor=2, mode='nearest')]
        current_channels = int(in_channels)
        for _ in range(max(1, int(blocks_count))):
            layers.extend(list(_conv_bn_lrelu(current_channels, out_channels)))
            current_channels = int(out_channels)
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class _UNetPlusPlusNode(nn.Module):
    def __init__(self, in_channels, out_channels, *, gn_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(in_channels, out_channels, norm='gn', act='leaky', pooling=False, gn_groups=gn_groups),
            ConvBlock(out_channels, out_channels, norm='gn', act='leaky', pooling=False, gn_groups=gn_groups),
        )

    def forward(self, *features):
        if not features:
            raise ValueError('UNet++ node expects at least one input tensor.')
        if len(features) == 1:
            return self.block(features[0])
        target_size = features[0].shape[-2:]
        merged = [features[0]]
        for feature in features[1:]:
            if feature.shape[-2:] != target_size:
                feature = F.interpolate(feature, size=target_size, mode='bilinear', align_corners=False)
            merged.append(feature)
        return self.block(torch.cat(merged, dim=1))
