"""CNN_Models: unet plus plus."""
from __future__ import annotations
from torch import nn
from neuralimage.model.NeuralNetwork.registrator import *
from neuralimage.model.NeuralNetwork.blocks import *

from neuralimage.model.NeuralNetwork.CNN_Models_parts.u_net_plus_plus import _UNetPlusPlusNode

@register_model('UNET++', model_type=ModelType.experimental)
class UnetPlusPlus(DeepSupervisionMixin, nn.Module):
    def __init__(self, in_channels: int = 1, deep_supervision: bool = False, base_channels: int = 32):
        super().__init__()
        self._init_deep_supervision(deep_supervision)
        c1 = max(16, int(base_channels))
        c2 = c1 * 2
        c3 = c1 * 4
        c4 = c1 * 8
        c5 = c1 * 16

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.x0_0 = _UNetPlusPlusNode(in_channels, c1)
        self.x1_0 = _UNetPlusPlusNode(c1, c2)
        self.x2_0 = _UNetPlusPlusNode(c2, c3)
        self.x3_0 = _UNetPlusPlusNode(c3, c4)
        self.x4_0 = _UNetPlusPlusNode(c4, c5)

        self.x0_1 = _UNetPlusPlusNode(c1 + c2, c1)
        self.x1_1 = _UNetPlusPlusNode(c2 + c3, c2)
        self.x2_1 = _UNetPlusPlusNode(c3 + c4, c3)
        self.x3_1 = _UNetPlusPlusNode(c4 + c5, c4)

        self.x0_2 = _UNetPlusPlusNode((c1 * 2) + c2, c1)
        self.x1_2 = _UNetPlusPlusNode((c2 * 2) + c3, c2)
        self.x2_2 = _UNetPlusPlusNode((c3 * 2) + c4, c3)

        self.x0_3 = _UNetPlusPlusNode((c1 * 3) + c2, c1)
        self.x1_3 = _UNetPlusPlusNode((c2 * 3) + c3, c2)

        self.x0_4 = _UNetPlusPlusNode((c1 * 4) + c2, c1)

        self.output_heads = SegmentationHeadBundle(c1, aux_channels=(c1, c1, c1))
        self.final_head = self.output_heads.primary
        self.confidence_head = self.output_heads.confidence
        self.ds_head3, self.ds_head2, self.ds_head1 = self.output_heads.auxiliary

    def forward(self, x):
        x0_0 = self.x0_0(x)
        x1_0 = self.x1_0(self.pool(x0_0))
        x2_0 = self.x2_0(self.pool(x1_0))
        x3_0 = self.x3_0(self.pool(x2_0))
        x4_0 = self.x4_0(self.pool(x3_0))

        x0_1 = self.x0_1(x0_0, x1_0)
        x1_1 = self.x1_1(x1_0, x2_0)
        x2_1 = self.x2_1(x2_0, x3_0)
        x3_1 = self.x3_1(x3_0, x4_0)

        x0_2 = self.x0_2(x0_0, x0_1, x1_1)
        x1_2 = self.x1_2(x1_0, x1_1, x2_1)
        x2_2 = self.x2_2(x2_0, x2_1, x3_1)

        x0_3 = self.x0_3(x0_0, x0_1, x0_2, x1_2)
        x1_3 = self.x1_3(x1_0, x1_1, x1_2, x2_2)

        x0_4 = self.x0_4(x0_0, x0_1, x0_2, x0_3, x1_3)

        primary, confidence, auxiliary_outputs = self.output_heads(x0_4, (x0_3, x0_2, x0_1))
        return self._build_model_outputs(
            primary,
            auxiliary_outputs=auxiliary_outputs,
            confidence=confidence,
        )
