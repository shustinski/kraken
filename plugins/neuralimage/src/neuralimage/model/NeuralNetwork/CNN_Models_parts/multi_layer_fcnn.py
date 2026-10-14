"""CNN_Models: multi layer fcnn."""
from __future__ import annotations
from torch import nn
from neuralimage.model.NeuralNetwork.registrator import *
from neuralimage.model.NeuralNetwork.blocks import *

from neuralimage.model.NeuralNetwork.CNN_Models_parts.u_net_plus_plus import (
    _FCNNDecoderStage,
    _FCNNEncoderStage,
)

class MultiLayerFCNN(DeepSupervisionMixin, nn.Module):
    def __init__(self, input_channels, layers, start_filter, step_filter, deep_supervision: bool = False):
        super(MultiLayerFCNN, self).__init__()
        self._init_deep_supervision(deep_supervision)
        encoder_filters = [int(start_filter) + int(step_filter) * idx for idx in range(int(layers) + 1)]
        if not encoder_filters:
            raise ValueError('MultiLayerFCNN requires at least one encoder stage.')

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        in_channels = int(input_channels)
        for idx, out_channels in enumerate(encoder_filters):
            self.encoders.append(_FCNNEncoderStage(in_channels, out_channels))
            if idx < len(encoder_filters) - 1:
                self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
            in_channels = out_channels

        bottleneck_channels = encoder_filters[-1] + int(step_filter)
        self.bottleneck = _FCNNEncoderStage(encoder_filters[-1], bottleneck_channels)

        decoder_filters = list(reversed(encoder_filters[:-1]))
        self.decoders = nn.ModuleList()
        current_channels = bottleneck_channels
        for out_channels in decoder_filters:
            self.decoders.append(_FCNNDecoderStage(current_channels, out_channels, blocks_count=1))
            current_channels = out_channels

        aux_channels = tuple(reversed(decoder_filters[-3:]))
        self.output_heads = SegmentationHeadBundle(
            current_channels,
            aux_channels=aux_channels,
            primary_kernel_size=3,
        )
        self.output_layer = self.output_heads.primary
        self.confidence_head = self.output_heads.confidence
        self.deep_supervision_heads = self.output_heads.auxiliary

    def forward(self, x):
        for idx, layer in enumerate(self.encoders):
            x = layer(x)
            if idx < len(self.pools):
                x = self.pools[idx](x)

        x = self.bottleneck(x)

        decoder_features = []
        for layer in self.decoders:
            x = layer(x)
            decoder_features.append(x)

        selected_features = tuple(reversed(decoder_features[-len(self.deep_supervision_heads):]))
        primary, confidence, auxiliary_outputs = self.output_heads(x, selected_features)
        return self._build_model_outputs(primary, auxiliary_outputs=auxiliary_outputs, confidence=confidence)
