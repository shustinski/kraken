from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Down, ResDSBlock, Up
from .context_utils import normalize_channel_sequence, normalize_size_pair
from .registrator import register_model


def _resolve_group_norm_groups(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, int(channels)), 0, -1):
        if int(channels) % groups == 0:
            return groups
    return 1


class _ContextEncoder(nn.Module):
    """Lightweight encoder used for the reduced-resolution context branch."""

    def __init__(
        self,
        in_ch: int,
        channels: tuple[int, ...],
        *,
        dropout: float = 0.03,
    ) -> None:
        super().__init__()
        self.stem = ResDSBlock(in_ch, channels[0], gn_groups=_resolve_group_norm_groups(channels[0]), dropout=0.0)
        blocks: list[nn.Module] = []
        for in_channels, out_channels in zip(channels[:-1], channels[1:]):
            blocks.append(
                Down(
                    in_channels,
                    out_channels,
                    gn_groups=_resolve_group_norm_groups(out_channels),
                    dropout=dropout,
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return x


@register_model('FrameUnet')
class QuasiDualScaleUNet(nn.Module):
    """U-Net with a lightweight context encoder fused at the local bottleneck."""

    def __init__(
        self,
        in_ch: int = 1,
        *,
        local_crop_size: tuple[int, int] | None = None,
        context_crop_size: tuple[int, int] | None = None,
        context_input_size: tuple[int, int] | None = None,
        context_branch_channels: tuple[int, ...] | int | None = (16, 32, 64, 128),
        fusion_type: str = 'concat',
        use_context_branch: bool = True,
        local_base_channels: int = 32,
        dropout_stem: float = 0.0,
        dropout_down: float = 0.08,
        dropout_bottleneck: float = 0.15,
        dropout_up: float = 0.03,
    ) -> None:
        super().__init__()
        base_size = normalize_size_pair(local_crop_size, fallback=(256, 256))
        self.local_crop_size = base_size
        self.context_crop_size = normalize_size_pair(
            context_crop_size,
            fallback=(base_size[0] * 2, base_size[1] * 2),
        )
        self.context_input_size = normalize_size_pair(
            context_input_size,
            fallback=base_size,
        )
        self.context_branch_channels = normalize_channel_sequence(
            context_branch_channels,
            fallback=(16, 32, 64, 128),
        )
        self.fusion_type = str(fusion_type or 'concat').strip().lower()
        if self.fusion_type != 'concat':
            raise ValueError(f'Unsupported fusion_type: {fusion_type!r}. Only "concat" is supported.')
        self.use_context_branch = bool(use_context_branch)

        c1 = max(8, int(local_base_channels))
        c2 = c1 * 2
        c3 = c1 * 4
        c4 = c1 * 8
        c5 = c1 * 16

        self.stem = ResDSBlock(in_ch, c1, gn_groups=_resolve_group_norm_groups(c1), dropout=dropout_stem)
        self.down1 = Down(c1, c2, gn_groups=_resolve_group_norm_groups(c2), dropout=dropout_down)
        self.down2 = Down(c2, c3, gn_groups=_resolve_group_norm_groups(c3), dropout=dropout_down)
        self.down3 = Down(c3, c4, gn_groups=_resolve_group_norm_groups(c4), dropout=dropout_down)
        self.down4 = Down(c4, c5, gn_groups=_resolve_group_norm_groups(c5), dropout=dropout_bottleneck)

        self.context_encoder = (
            _ContextEncoder(
                in_ch,
                self.context_branch_channels,
                dropout=dropout_up,
            )
            if self.use_context_branch
            else None
        )
        context_channels = int(self.context_branch_channels[-1]) if self.use_context_branch else 0
        self.fusion = (
            nn.Sequential(
                nn.Conv2d(c5 + context_channels, c5, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(_resolve_group_norm_groups(c5), c5),
                nn.SiLU(inplace=True),
            )
            if self.use_context_branch
            else nn.Identity()
        )

        self.up4 = Up(c5, c4, c4, gn_groups=_resolve_group_norm_groups(c4), dropout=dropout_up)
        self.up3 = Up(c4, c3, c3, gn_groups=_resolve_group_norm_groups(c3), dropout=dropout_up)
        self.up2 = Up(c3, c2, c2, gn_groups=_resolve_group_norm_groups(c2), dropout=dropout_up)
        self.up1 = Up(c2, c1, c1, gn_groups=_resolve_group_norm_groups(c1), dropout=dropout_up)
        self.head = nn.Conv2d(c1, 1, kernel_size=1)

    @staticmethod
    def _extract_inputs(
        local_x: torch.Tensor | Mapping[str, torch.Tensor],
        context_x: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(local_x, Mapping):
            local_tensor = local_x.get('local_image')
            if local_tensor is None:
                raise KeyError('Expected "local_image" in dual-scale input mapping.')
            context_tensor = local_x.get('context_image')
            return local_tensor, context_tensor
        return local_x, context_x

    def forward(
        self,
        local_x: torch.Tensor | Mapping[str, torch.Tensor],
        context_x: torch.Tensor | None = None,
    ) -> torch.Tensor:
        local_x, context_x = self._extract_inputs(local_x, context_x)

        s1 = self.stem(local_x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        bottleneck = self.down4(s4)

        if self.use_context_branch:
            if context_x is None:
                context_x = F.interpolate(
                    local_x,
                    size=(int(self.context_input_size[1]), int(self.context_input_size[0])),
                    mode='bilinear',
                    align_corners=False,
                )
            context_features = self.context_encoder(context_x)
            if context_features.shape[-2:] != bottleneck.shape[-2:]:
                context_features = F.adaptive_avg_pool2d(context_features, output_size=bottleneck.shape[-2:])
            bottleneck = self.fusion(torch.cat([bottleneck, context_features], dim=1))

        x = self.up4(bottleneck, s4)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        return self.head(x)


@register_model('UNetWithContextBranch')
class UNetWithContextBranch(QuasiDualScaleUNet):
    """Alias for QuasiDualScaleUNet used by older or descriptive configs."""

    pass
