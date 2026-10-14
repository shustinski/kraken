"""CNN_Models: small fcnn."""
from __future__ import annotations
from torch import nn
from neuralimage.model.NeuralNetwork.registrator import *
from neuralimage.model.NeuralNetwork.blocks import *

from neuralimage.model.NeuralNetwork.CNN_Models_parts.u_net_plus_plus import _FCNNDecoderStage

@register_model(name='S 660k', model_type=ModelType.deprecated)
class SmallFCNN(DeepSupervisionMixin, nn.Module):
    def __init__(self, input_channels = 1, deep_supervision: bool = False):
        super(SmallFCNN, self).__init__()
        self._init_deep_supervision(deep_supervision)

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, stride=2, padding=0),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, stride=2, padding=0),

            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.2),

            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.2),

            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, stride=2, padding=0)
        )

        self.decoder_stage3 = _FCNNDecoderStage(96, 96, blocks_count=3)
        self.decoder_stage2 = _FCNNDecoderStage(96, 64, blocks_count=2)
        self.decoder_stage1 = _FCNNDecoderStage(64, 32, blocks_count=2)
        self.output_heads = SegmentationHeadBundle(32, aux_channels=(64, 96), primary_kernel_size=3)
        self.output_layer = self.output_heads.primary
        self.confidence_head = self.output_heads.confidence
        self.ds_stage2_head, self.ds_stage3_head = self.output_heads.auxiliary

    def forward(self, x):
        x = self.encoder(x)
        d3 = self.decoder_stage3(x)
        d2 = self.decoder_stage2(d3)
        d1 = self.decoder_stage1(d2)
        primary, confidence, auxiliary_outputs = self.output_heads(d1, (d2, d3))
        return self._build_model_outputs(
            primary,
            auxiliary_outputs=auxiliary_outputs,
            confidence=confidence,
        )
