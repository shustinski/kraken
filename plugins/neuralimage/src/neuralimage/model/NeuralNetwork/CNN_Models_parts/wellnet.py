"""CNN_Models: wellnet."""
from __future__ import annotations
from torch import nn
from neuralimage.model.NeuralNetwork.registrator import *
from neuralimage.model.NeuralNetwork.blocks import *

@register_model('Wellnet 86.5M', model_type=ModelType.deprecated)
class Wellnet(DeepSupervisionMixin, nn.Module):
    def __init__(self,in_channels: int = 1, deep_supervision: bool = False):
        super().__init__()
        self._init_deep_supervision(deep_supervision)

        self.start_conv = build_conv_sequence(
            (
                (in_channels, 64, 0.0, False),
                (64, 128, 0.0, True),
            ),
            norm='bn',
            act='leaky',
        )

        self.e1 = ConvBlock(128, 256, norm='bn', act='leaky', pooling=True)
        self.e2 = ConvBlock(256, 512, norm='bn', act='leaky', pooling=True)
        self.e3 = ConvBlock(512, 1024, norm='bn', act='leaky', pooling=True)

        self.b1 = Bottleneck(1024)

        self.d1 = ConvTransposeEncoder(2048, 1024, 512)
        self.d2 = ConvTransposeEncoder(512, 512, 256)
        self.d3 = ConvTransposeEncoder(256, 256, 128)

        self.finish_upsample = nn.Sequential(
            ConvTransposeEncoder(128, 0, 64),
            nn.Conv2d(64, 1, kernel_size=1)
        )
        self.output_heads = SegmentationHeadBundle(64, aux_channels=(128, 256, 512))
        self.confidence_head = self.output_heads.confidence
        self.ds_d3_head, self.ds_d2_head, self.ds_d1_head = self.output_heads.auxiliary


    def forward(self, x):

        x = self.start_conv(x)

        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)

        b = self.b1(e3)

        d1 = self.d1(b, e3)
        d2 = self.d2(d1, e2)
        d3 = self.d3(d2, e1)

        final_features = self.finish_upsample[0](d3)
        primary = self.finish_upsample[1](final_features)
        confidence = self.confidence_head(final_features)
        auxiliary_outputs = tuple(head(feature) for head, feature in zip(self.output_heads.auxiliary, (d3, d2, d1)))
        return self._build_model_outputs(
            primary,
            auxiliary_outputs=auxiliary_outputs,
            confidence=confidence,
        )
