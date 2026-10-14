"""CNN_Models: unet."""
from __future__ import annotations
from torch import nn
from neuralimage.model.NeuralNetwork.registrator import *
from neuralimage.model.NeuralNetwork.blocks import *

@register_model('Unet 21.6M', model_type=ModelType.deprecated)
class Unet(DeepSupervisionMixin, nn.Module):
    def __init__(self,in_channels: int = 1, deep_supervision: bool = False):
        super().__init__()
        self._init_deep_supervision(deep_supervision)

        self.e1 = ConvBlock(in_channels, 64, norm='bn', act='leaky', pooling=True)
        self.e2 = ConvBlock(64,  128, norm='bn', act='leaky', pooling=True)
        self.e3 = ConvBlock(128, 256, norm='bn', act='leaky', pooling=True)
        self.e4 = ConvBlock(256, 512, norm='bn', act='leaky', pooling=True)

        self.b1 = Bottleneck(512)

        self.d1 = ConvTransposeEncoder(1024, 512, 256)
        self.d2 = ConvTransposeEncoder(256, 256, 128)
        self.d3 = ConvTransposeEncoder(128, 128, 64)
        self.d4 = ConvTransposeEncoder(64, 64, 64)

        self.output_heads = SegmentationHeadBundle(64, aux_channels=(64, 64, 128))
        self.out_conv = self.output_heads.primary
        self.confidence_head = self.output_heads.confidence
        self.ds_d4_head, self.ds_d3_head, self.ds_d2_head = self.output_heads.auxiliary


    def forward(self, x):

        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)

        b = self.b1(e4)

        d1 = self.d1(b, e4)
        d2 = self.d2(d1, e3)
        d3 = self.d3(d2, e2)
        d4 = self.d4(d3, e1)

        primary, confidence, auxiliary_outputs = self.output_heads(d4, (d4, d3, d2))
        return self._build_model_outputs(
            primary,
            auxiliary_outputs=auxiliary_outputs,
            confidence=confidence,
        )
