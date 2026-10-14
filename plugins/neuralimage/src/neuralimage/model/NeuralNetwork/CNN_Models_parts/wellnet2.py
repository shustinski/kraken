"""CNN_Models: wellnet2."""
from __future__ import annotations
from torch import nn
from neuralimage.model.NeuralNetwork.registrator import *
from neuralimage.model.NeuralNetwork.blocks import *

@register_model('Wellnet2', model_type=ModelType.deprecated)
class Wellnet2(DeepSupervisionMixin, nn.Module):
    def __init__(self,in_channels: int = 1, deep_supervision: bool = False):
        super().__init__()
        self._init_deep_supervision(deep_supervision)

        self.start_conv = build_conv_sequence(
            (
                (in_channels, 64, 0.1, False),
                (64, 64, 0.1, False),
                (64, 64, 0.1, True),
            ),
            norm='bn',
            act='leaky',
        )

        self.e1 = build_conv_sequence(
            (
                (64, 128, 0.2, False),
                (128, 128, 0.2, False),
                (128, 128, 0.2, True),
            ),
            norm='bn',
            act='leaky',
        )

        self.e2 = build_conv_sequence(
            (
                (128, 256, 0.3, False),
                (256, 256, 0.3, False),
                (256, 256, 0.3, True),
            ),
            norm='bn',
            act='leaky',
        )
        self.e3 = build_conv_sequence(
            (
                (256, 512, 0.3, False),
                (512, 512, 0.3, False),
                (512, 512, 0.3, True),
            ),
            norm='bn',
            act='leaky',
        )

        self.b1 = Bottleneck(512)

        self.d1 = ConvTransposeEncoder(1024, 512, 512, norm='bn', act='leaky', droput=0.3)
        self.d2 = ConvTransposeEncoder(512, 256, 256, norm='bn', act='leaky', droput=0.2)
        self.d3 = ConvTransposeEncoder(256, 128, 128, norm='bn', act='leaky', droput=0.1)

        self.finish_upsample = nn.Sequential(
            ConvTransposeEncoder(128, 0, 64, norm='bn', act='leaky', droput=0.1),
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
