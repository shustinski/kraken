from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import DeepSupervisionMixin, Down, ResDSBlock, Up
from .context_utils import normalize_channel_sequence, normalize_size_pair
from .registrator import ModelType, register_model


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


class CoordinateAwareCrossAttention(nn.Module):
    """Cross-attention from local bottleneck tokens to the full global feature map."""

    def __init__(
        self,
        *,
        local_channels: int,
        global_channels: int,
        attention_dim: int,
        num_heads: int,
        max_global_tokens: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        resolved_dim = max(8, int(attention_dim))
        resolved_heads = max(1, min(int(num_heads), resolved_dim))
        while resolved_dim % resolved_heads != 0 and resolved_heads > 1:
            resolved_heads -= 1

        self.local_channels = int(local_channels)
        self.global_channels = int(global_channels)
        self.attention_dim = int(resolved_dim)
        self.num_heads = int(resolved_heads)
        self.max_global_tokens = max(1, int(max_global_tokens))

        self.local_query = nn.Conv2d(self.local_channels, self.attention_dim, kernel_size=1, bias=False)
        self.global_key_value = nn.Conv2d(self.global_channels, self.attention_dim, kernel_size=1, bias=False)
        self.coord_mlp = nn.Sequential(
            nn.Linear(4, self.attention_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.attention_dim, self.attention_dim),
        )
        self.query_norm = nn.LayerNorm(self.attention_dim)
        self.key_value_norm = nn.LayerNorm(self.attention_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.attention_dim,
            num_heads=self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.output_projection = nn.Linear(self.attention_dim, self.local_channels)

    def _downsample_global_features(self, global_feat: torch.Tensor) -> torch.Tensor:
        tokens = int(global_feat.shape[-2] * global_feat.shape[-1])
        if tokens <= self.max_global_tokens:
            return global_feat

        height, width = int(global_feat.shape[-2]), int(global_feat.shape[-1])
        scale = math.sqrt(float(self.max_global_tokens) / float(max(1, tokens)))
        target_h = max(1, int(math.floor(height * scale)))
        target_w = max(1, int(math.floor(width * scale)))
        while target_h * target_w > self.max_global_tokens:
            if target_h >= target_w and target_h > 1:
                target_h -= 1
            elif target_w > 1:
                target_w -= 1
            else:
                break
        # The global branch must represent the full frame, but attention cost is
        # O(local_tokens * global_tokens). Adaptive pooling keeps all regions
        # represented while bounding memory and latency.
        return F.adaptive_avg_pool2d(global_feat, output_size=(target_h, target_w))

    @staticmethod
    def _normalize_coords_for_batch(
        patch_coords_norm: torch.Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if patch_coords_norm is None:
            return torch.zeros((batch_size, 4), device=device, dtype=dtype)
        coords = patch_coords_norm.to(device=device, dtype=dtype)
        if coords.ndim == 1:
            coords = coords.view(1, 4)
        if coords.ndim != 2 or int(coords.shape[-1]) != 4:
            raise ValueError(f'patch_coords_norm must have shape [B, 4], got {tuple(coords.shape)!r}.')
        if int(coords.shape[0]) == 1 and batch_size > 1:
            coords = coords.expand(batch_size, -1)
        if int(coords.shape[0]) != batch_size:
            raise ValueError(
                f'patch_coords_norm batch mismatch: expected {batch_size}, got {int(coords.shape[0])}.'
            )
        return torch.clamp(coords, 0.0, 1.0)

    def forward(
        self,
        local_feat: torch.Tensor,
        global_feat: torch.Tensor,
        patch_coords_norm: torch.Tensor | None,
    ) -> torch.Tensor:
        if local_feat.ndim != 4 or global_feat.ndim != 4:
            raise ValueError('Cross-attention expects local/global feature maps in BCHW format.')
        batch_size, _channels, local_h, local_w = local_feat.shape
        if int(global_feat.shape[0]) == 1 and batch_size > 1:
            global_feat = global_feat.expand(batch_size, -1, -1, -1)
        if int(global_feat.shape[0]) != batch_size:
            raise ValueError(
                f'Global feature batch mismatch: expected {batch_size}, got {int(global_feat.shape[0])}.'
            )

        global_feat = self._downsample_global_features(global_feat)
        query = self.local_query(local_feat)
        key_value = self.global_key_value(global_feat)

        query_tokens = query.flatten(2).transpose(1, 2)
        global_tokens = key_value.flatten(2).transpose(1, 2)
        coords = self._normalize_coords_for_batch(
            patch_coords_norm,
            batch_size=batch_size,
            device=local_feat.device,
            dtype=query_tokens.dtype,
        )
        query_tokens = query_tokens + self.coord_mlp(coords).unsqueeze(1)
        query_tokens = self.query_norm(query_tokens)
        global_tokens = self.key_value_norm(global_tokens)

        attended_tokens, _ = self.attention(
            query=query_tokens,
            key=global_tokens,
            value=global_tokens,
            need_weights=False,
        )
        attended_tokens = self.output_projection(attended_tokens)
        return (
            attended_tokens
            .reshape(batch_size, local_h, local_w, self.local_channels)
            .permute(0, 3, 1, 2)
            .contiguous()
        )


@register_model('UNetWithContextBranch', model_type=ModelType.experimental)
@register_model('quasi_dual_scale_unet', model_type=ModelType.experimental)
@register_model('FrameUnet', model_type=ModelType.experimental)
class QuasiDualScaleUNet(DeepSupervisionMixin, nn.Module):
    """U-Net with full-frame global context fused at the local bottleneck."""

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
        use_cross_attention: bool = True,
        attention_dim: int | None = None,
        attention_heads: int = 4,
        attention_max_global_tokens: int = 1024,
        deep_supervision: bool = False,
        local_base_channels: int = 32,
        dropout_stem: float = 0.0,
        dropout_down: float = 0.08,
        dropout_bottleneck: float = 0.15,
        dropout_up: float = 0.03,
    ) -> None:
        super().__init__()
        self._init_deep_supervision(deep_supervision)
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
        if self.fusion_type not in {'concat', 'add'}:
            raise ValueError(f'Unsupported fusion_type: {fusion_type!r}. Expected "concat" or "add".')
        self.use_context_branch = bool(use_context_branch)
        self.use_cross_attention = bool(self.use_context_branch and use_cross_attention)
        self.attention_dim = (
            max(8, int(attention_dim))
            if attention_dim is not None
            else min(128, max(8, int(local_base_channels) * 4))
        )
        self.attention_heads = max(1, int(attention_heads))
        self.attention_max_global_tokens = max(1, int(attention_max_global_tokens))

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
        self.cross_attention = (
            CoordinateAwareCrossAttention(
                local_channels=c5,
                global_channels=context_channels,
                attention_dim=self.attention_dim,
                num_heads=self.attention_heads,
                max_global_tokens=self.attention_max_global_tokens,
                dropout=dropout_up,
            )
            if self.use_cross_attention
            else None
        )
        if not self.use_context_branch:
            self.fusion = nn.Identity()
            self.legacy_context_projection = None
        elif self.use_cross_attention:
            self.legacy_context_projection = None
            self.fusion = (
                nn.Sequential(
                    nn.Conv2d(c5 * 2, c5, kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(_resolve_group_norm_groups(c5), c5),
                    nn.SiLU(inplace=True),
                )
                if self.fusion_type == 'concat'
                else nn.Identity()
            )
        elif self.fusion_type == 'concat':
            self.legacy_context_projection = None
            self.fusion = nn.Sequential(
                nn.Conv2d(c5 + context_channels, c5, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(_resolve_group_norm_groups(c5), c5),
                nn.SiLU(inplace=True),
            )
        else:
            self.legacy_context_projection = nn.Conv2d(context_channels, c5, kernel_size=1, bias=False)
            self.fusion = nn.Identity()

        self.up4 = Up(c5, c4, c4, gn_groups=_resolve_group_norm_groups(c4), dropout=dropout_up)
        self.up3 = Up(c4, c3, c3, gn_groups=_resolve_group_norm_groups(c3), dropout=dropout_up)
        self.up2 = Up(c3, c2, c2, gn_groups=_resolve_group_norm_groups(c2), dropout=dropout_up)
        self.up1 = Up(c2, c1, c1, gn_groups=_resolve_group_norm_groups(c1), dropout=dropout_up)
        self.head = nn.Conv2d(c1, 1, kernel_size=1)
        self.confidence_head = nn.Conv2d(c1, 1, kernel_size=1)
        self.ds_up2_head = nn.Conv2d(c2, 1, kernel_size=1)
        self.ds_up3_head = nn.Conv2d(c3, 1, kernel_size=1)
        self.ds_up4_head = nn.Conv2d(c4, 1, kernel_size=1)

    @staticmethod
    def _extract_inputs(
        local_x: torch.Tensor | Mapping[str, torch.Tensor],
        global_x: torch.Tensor | None,
        patch_coords_norm: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if isinstance(local_x, Mapping):
            local_tensor = local_x.get('local_image')
            if local_tensor is None:
                raise KeyError('Expected "local_image" in dual-scale input mapping.')
            global_tensor = local_x.get('global_image')
            if global_tensor is None:
                global_tensor = local_x.get('context_image')
            coords_tensor = local_x.get('patch_coords_norm')
            return local_tensor, global_tensor, coords_tensor
        return local_x, global_x, patch_coords_norm

    def _prepare_global_input(self, local_x: torch.Tensor, global_x: torch.Tensor | None) -> torch.Tensor:
        if global_x is None:
            return F.interpolate(
                local_x,
                size=(int(self.context_input_size[1]), int(self.context_input_size[0])),
                mode='bilinear',
                align_corners=False,
            )
        if global_x.ndim == 3:
            global_x = global_x.unsqueeze(0)
        if global_x.ndim != 4:
            raise ValueError(f'Global/context image must be BCHW or CHW, got {tuple(global_x.shape)!r}.')
        if int(global_x.shape[0]) == 1 and int(local_x.shape[0]) > 1:
            global_x = global_x.expand(int(local_x.shape[0]), -1, -1, -1)
        if int(global_x.shape[0]) != int(local_x.shape[0]):
            raise ValueError(
                f'Global/context image batch mismatch: expected {int(local_x.shape[0])}, '
                f'got {int(global_x.shape[0])}.'
            )
        return global_x

    def forward(
        self,
        local_x: torch.Tensor | Mapping[str, torch.Tensor],
        global_x: torch.Tensor | None = None,
        patch_coords_norm: torch.Tensor | None = None,
    ) -> torch.Tensor:
        local_x, global_x, patch_coords_norm = self._extract_inputs(local_x, global_x, patch_coords_norm)

        s1 = self.stem(local_x)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        bottleneck = self.down4(s4)

        if self.use_context_branch:
            global_x = self._prepare_global_input(local_x, global_x)
            global_features = self.context_encoder(global_x)
            if self.use_cross_attention:
                attended_global = self.cross_attention(bottleneck, global_features, patch_coords_norm)
                if self.fusion_type == 'concat':
                    bottleneck = self.fusion(torch.cat([bottleneck, attended_global], dim=1))
                else:
                    bottleneck = bottleneck + attended_global
            else:
                if global_features.shape[-2:] != bottleneck.shape[-2:]:
                    global_features = F.adaptive_avg_pool2d(global_features, output_size=bottleneck.shape[-2:])
                if self.fusion_type == 'concat':
                    bottleneck = self.fusion(torch.cat([bottleneck, global_features], dim=1))
                else:
                    bottleneck = bottleneck + self.legacy_context_projection(global_features)

        x = self.up4(bottleneck, s4)
        u4 = x
        x = self.up3(x, s3)
        u3 = x
        x = self.up2(x, s2)
        u2 = x
        x = self.up1(x, s1)
        primary = self.head(x)
        confidence = self.confidence_head(x)
        return self._build_model_outputs(
            primary,
            auxiliary_outputs=(
                self.ds_up2_head(u2),
                self.ds_up3_head(u3),
                self.ds_up4_head(u4),
            ),
            confidence=confidence,
        )
