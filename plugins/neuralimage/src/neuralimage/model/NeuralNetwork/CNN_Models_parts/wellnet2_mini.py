"""CNN_Models: wellnet2 mini."""
from __future__ import annotations
from torch import nn
from neuralimage.model.NeuralNetwork.registrator import *
from neuralimage.model.NeuralNetwork.blocks import *

@register_model('Wellnet2 mini', model_type=ModelType.deprecated)
class Wellnet2Mini(DeepSupervisionMixin, nn.Module):
    def __init__(self,in_channels: int = 1, deep_supervision: bool = False):
        super().__init__()
        self._init_deep_supervision(deep_supervision)
        base_channels = 32

        self.start_conv = build_conv_sequence(
            (
                (in_channels, base_channels, 0.1, False),
                (base_channels, base_channels, 0.1, False),
                (base_channels, base_channels, 0.1, True),
            ),
            norm='bn',
            act='leaky',
        )

        self.e1 = build_conv_sequence(
            (
                (base_channels, base_channels*2, 0.2, False),
                (base_channels*2, base_channels*2, 0.2, False),
                (base_channels*2, base_channels*2, 0.2, True),
            ),
            norm='bn',
            act='leaky',
        )

        self.e2 = build_conv_sequence(
            (
                (base_channels*2, base_channels*4, 0.3, False),
                (base_channels*4, base_channels*4, 0.3, False),
                (base_channels*4, base_channels*4, 0.3, True),
            ),
            norm='bn',
            act='leaky',
        )
        self.e3 = build_conv_sequence(
            (
                (base_channels*4, base_channels*8, 0.3, False),
                (base_channels*8, base_channels*8, 0.3, False),
                (base_channels*8, base_channels*8, 0.3, True),
            ),
            norm='bn',
            act='leaky',
        )

        self.b1 = Bottleneck(base_channels*8)

        self.d1 = ConvTransposeEncoder(base_channels*16, base_channels*8, base_channels*8, norm='bn', act='leaky', droput=0.3)
        self.d2 = ConvTransposeEncoder(base_channels*8, base_channels*4, base_channels*4, norm='bn', act='leaky', droput=0.2)
        self.d3 = ConvTransposeEncoder(base_channels*4, base_channels*2, base_channels*2, norm='bn', act='leaky', droput=0.1)

        self.finish_upsample = nn.Sequential(
            ConvTransposeEncoder(base_channels*2, 0, base_channels, norm='bn', act='leaky', droput=0.1),
            nn.Conv2d(base_channels, 1, kernel_size=1)
        )
        self.output_heads = SegmentationHeadBundle(base_channels, aux_channels=(base_channels*2, base_channels*4, base_channels*8))
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
