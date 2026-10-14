"""CNN_Models: efficient u net base."""
from __future__ import annotations
from collections.abc import Sequence
from torch import nn
from neuralimage.model.NeuralNetwork.registrator import *
from neuralimage.model.NeuralNetwork.blocks import *
from neuralimage.heads.multi_target import MultiTargetHeadBundle

class _EfficientUNetBase(DeepSupervisionMixin, nn.Module):
    def __init__(
        self,
        *,
        in_ch: int = 1,
        base_ch: int,
        deep_supervision: bool = False,
        dropout_stem: float = 0.0,
        dropout_down: float = 0.08,
        dropout_bottleneck: float = 0.15,
        dropout_up: float = 0.03,
        gn_groups: int = 8,
        supervision_heads: tuple[str, ...] | Sequence[str] = (),
    ) -> None:
        super().__init__()
        self._init_deep_supervision(deep_supervision)
        self._supervision_head_names = tuple(str(name) for name in supervision_heads)
        c1 = int(base_ch)
        c2 = c1 * 2
        c3 = c1 * 4
        c4 = c1 * 8
        c5 = c1 * 16

        self.stem = ResDSBlock(in_ch, c1, gn_groups=gn_groups, dropout=dropout_stem)
        self.down1 = Down(c1, c2, gn_groups=gn_groups, dropout=dropout_down)
        self.down2 = Down(c2, c3, gn_groups=gn_groups, dropout=dropout_down)
        self.down3 = Down(c3, c4, gn_groups=gn_groups, dropout=dropout_down)
        self.down4 = Down(c4, c5, gn_groups=gn_groups, dropout=dropout_bottleneck)

        self.up4 = Up(c5, c4, c4, gn_groups=gn_groups, dropout=dropout_up)
        self.up3 = Up(c4, c3, c3, gn_groups=gn_groups, dropout=dropout_up)
        self.up2 = Up(c3, c2, c2, gn_groups=gn_groups, dropout=dropout_up)
        self.up1 = Up(c2, c1, c1, gn_groups=gn_groups, dropout=dropout_up)

        if self._supervision_head_names:
            self.output_heads = MultiTargetHeadBundle(
                c1,
                supervision_heads=self._supervision_head_names,
                aux_channels=(c2, c3, c4),
            )
        else:
            self.output_heads = SegmentationHeadBundle(c1, aux_channels=(c2, c3, c4))
        self.head = self.output_heads.primary
        self.confidence_head = self.output_heads.confidence
        self.ds_up2_head, self.ds_up3_head, self.ds_up4_head = self.output_heads.auxiliary

    def forward(self, x):
        s1 = self.stem(x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        b = self.down4(s4)

        x = self.up4(b, s4)
        u4 = x
        x = self.up3(x, s3)
        u3 = x
        x = self.up2(x, s2)
        u2 = x
        x = self.up1(x, s1)
        if self._supervision_head_names:
            primary, confidence, auxiliary_outputs, supervision_outputs = self.output_heads(
                x,
                (u2, u3, u4),
                include_supervision=self.training,
            )
            return self._build_model_outputs(
                primary,
                auxiliary_outputs=auxiliary_outputs,
                confidence=confidence,
                supervision_outputs=supervision_outputs if self.training else None,
            )
        primary, confidence, auxiliary_outputs = self.output_heads(x, (u2, u3, u4))
        return self._build_model_outputs(
            primary,
            auxiliary_outputs=auxiliary_outputs,
            confidence=confidence,
        )
