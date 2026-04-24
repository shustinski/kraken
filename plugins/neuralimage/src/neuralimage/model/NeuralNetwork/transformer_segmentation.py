from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    import timm
except Exception:  # pragma: no cover - dependency availability is validated by integration tests.
    timm = None

from .registrator import ModelType, register_model


def _env_flag(name: str, default: bool = False) -> bool:
    raw = str(os.getenv(name, '1' if default else '0')).strip().lower()
    return raw in {'1', 'true', 'yes', 'on'}


def _allow_pretrained_weights() -> bool:
    return _env_flag('NEURALIMAGE_SWIN_PRETRAINED', False)


def _resolve_project_root() -> Path:
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[4]


def _resolve_internal_timm_weights_dir() -> Path:
    return _resolve_project_root() / 'resources' / 'internal' / 'models' / 'timm'


def _resolve_local_swin_weight_file(model_name: str) -> Path:
    return _resolve_internal_timm_weights_dir() / str(model_name).strip() / 'model.safetensors'


def _group_norm_2d(channels: int, *, max_groups: int = 32) -> nn.GroupNorm:
    resolved_channels = int(channels)
    for groups in range(min(max_groups, resolved_channels), 0, -1):
        if resolved_channels % groups == 0:
            return nn.GroupNorm(groups, resolved_channels)
    return nn.GroupNorm(1, resolved_channels)


def _enable_grad_checkpointing_if_supported(backbone: nn.Module, enabled: bool) -> None:
    if not enabled:
        return
    set_fn = getattr(backbone, 'set_grad_checkpointing', None)
    if callable(set_fn):
        set_fn(True)


def _freeze_module(module: nn.Module, enabled: bool) -> None:
    if not enabled:
        return
    for parameter in module.parameters():
        parameter.requires_grad = False


def _resolve_backbone_input_hw(backbone: nn.Module) -> tuple[int, int] | None:
    expected_hw = getattr(backbone, 'expected_input_hw', None)
    if isinstance(expected_hw, int):
        return int(expected_hw), int(expected_hw)
    if isinstance(expected_hw, (tuple, list)) and len(expected_hw) == 2:
        h, w = expected_hw
        if isinstance(h, int) and isinstance(w, int):
            return int(h), int(w)
    return None


def _run_backbone_with_adaptive_input(backbone: nn.Module, x: torch.Tensor) -> Sequence[torch.Tensor]:
    expected_hw = _resolve_backbone_input_hw(backbone)
    if expected_hw is not None and tuple(int(v) for v in x.shape[-2:]) != expected_hw:
        x = F.interpolate(x, size=expected_hw, mode='bilinear', align_corners=False)
    return backbone(x)


def _normalize_backbone_features(
    features: Sequence[torch.Tensor],
    expected_channels: Sequence[int],
) -> list[torch.Tensor]:
    if len(features) != len(expected_channels):
        raise RuntimeError(
            f'Unexpected number of backbone features: expected {len(expected_channels)}, got {len(features)}'
        )
    normalized: list[torch.Tensor] = []
    for idx, feature in enumerate(features):
        if feature.ndim != 4:
            raise RuntimeError(f'Backbone feature[{idx}] must be 4D, got shape={tuple(feature.shape)}')
        expected_c = int(expected_channels[idx])
        if int(feature.shape[1]) == expected_c:
            normalized.append(feature)
            continue
        if int(feature.shape[-1]) == expected_c:
            normalized.append(feature.permute(0, 3, 1, 2).contiguous())
            continue
        raise RuntimeError(
            (
                f'Unexpected channels for backbone feature[{idx}]: '
                f'expected {expected_c}, got shape={tuple(feature.shape)}'
            )
        )
    return normalized


class FeatureInfo:
    """Pickle-safe feature metadata container used by native backbones."""

    def __init__(self, channels: Sequence[int]) -> None:
        self._channels = [int(v) for v in channels]

    def channels(self) -> list[int]:
        return list(self._channels)


class _ConvResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.block(x) + x)


class _ConvStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, downsample: bool, depth: int) -> None:
        super().__init__()
        stride = 2 if downsample else 1
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[_ConvResidualBlock(out_channels) for _ in range(max(1, int(depth)))])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return self.blocks(x)


class NativeHierarchicalBackbone(nn.Module):
    """Native PyTorch multi-scale backbone."""

    def __init__(
        self,
        input_channels: int,
        *,
        feature_channels: Sequence[int],
        stage_depths: Sequence[int],
        expected_input_hw: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        if len(feature_channels) != 4:
            raise ValueError('feature_channels must contain 4 entries.')
        if len(stage_depths) != 4:
            raise ValueError('stage_depths must contain 4 entries.')

        c1, c2, c3, c4 = [int(v) for v in feature_channels]
        d1, d2, d3, d4 = [int(v) for v in stage_depths]
        self.expected_input_hw = expected_input_hw
        self._use_grad_checkpointing = False
        self.feature_info = FeatureInfo([c1, c2, c3, c4])

        self.stem = nn.Sequential(
            nn.Conv2d(int(input_channels), c1, kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm2d(c1),
            nn.GELU(),
        )
        self.stage1 = _ConvStage(c1, c1, downsample=False, depth=d1)
        self.stage2 = _ConvStage(c1, c2, downsample=True, depth=d2)
        self.stage3 = _ConvStage(c2, c3, downsample=True, depth=d3)
        self.stage4 = _ConvStage(c3, c4, downsample=True, depth=d4)

    def set_grad_checkpointing(self, enabled: bool = True) -> None:
        self._use_grad_checkpointing = bool(enabled)

    def _run_stage(self, stage: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self._use_grad_checkpointing and self.training and bool(x.requires_grad):
            try:
                # Reentrant mode is more stable across PyTorch builds for simple stage wrappers.
                return checkpoint(stage, x, use_reentrant=True)
            except Exception:
                # Fallback to regular forward to avoid hard failures from checkpoint internals.
                return stage(x)
        return stage(x)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        c1 = self._run_stage(self.stage1, x)
        c2 = self._run_stage(self.stage2, c1)
        c3 = self._run_stage(self.stage3, c2)
        c4 = self._run_stage(self.stage4, c3)
        return [c1, c2, c3, c4]


def _build_native_swin_backbone(backbone_name: str, input_channels: int) -> nn.Module:
    if timm is None:
        raise RuntimeError(
            'Real Swin backbones require the "timm" package. Install it to use Swin UPerNet and Mask2Former Swin.'
        )

    normalized_name = str(backbone_name).strip().lower()
    aliases = {
        'swin_b': 'swin_base_patch4_window7_224',
        'swin_base_patch4_window7_224': 'swin_base_patch4_window7_224',
        'swin_l': 'swin_large_patch4_window7_224',
        'swin_large_patch4_window7_224': 'swin_large_patch4_window7_224',
    }
    resolved_name = aliases.get(normalized_name)
    if resolved_name is None:
        raise ValueError(f'Unsupported Swin backbone name: {backbone_name!r}')

    create_kwargs: dict[str, object] = {
        'pretrained': False,
        'in_chans': int(input_channels),
        'features_only': True,
        'out_indices': (0, 1, 2, 3),
        'strict_img_size': False,
    }
    if _allow_pretrained_weights():
        local_weight_file = _resolve_local_swin_weight_file(resolved_name)
        if not local_weight_file.is_file():
            raise FileNotFoundError(
                'Offline Swin weights are enabled, but the local weight file is missing: '
                f'{local_weight_file}. Download the bundled timm weights into the internal folder first.'
            )
        create_kwargs['pretrained'] = True
        create_kwargs['pretrained_cfg_overlay'] = {'file': str(local_weight_file)}

    return timm.create_model(
        resolved_name,
        **create_kwargs,
    )


class PyramidPoolingModule(nn.Module):
    """PSP-style context aggregation used by UPerNet."""

    def __init__(self, in_channels: int, out_channels: int, pool_scales: Sequence[int] = (1, 2, 3, 6)) -> None:
        super().__init__()
        self.stages = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    _group_norm_2d(out_channels),
                    nn.GELU(),
                )
                for scale in pool_scales
            ]
        )
        merged_channels = in_channels + out_channels * len(pool_scales)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(merged_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _group_norm_2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        features = [x]
        for stage in self.stages:
            pooled = stage(x)
            pooled = F.interpolate(pooled, size=(h, w), mode='bilinear', align_corners=False)
            features.append(pooled)
        return self.bottleneck(torch.cat(features, dim=1))


class UPerNetHead(nn.Module):
    """Minimal UPerNet head for binary semantic segmentation."""

    def __init__(self, in_channels: Sequence[int], fpn_channels: int = 256, out_channels: int = 1) -> None:
        super().__init__()
        if len(in_channels) != 4:
            raise ValueError('UPerNetHead expects exactly 4 backbone feature maps.')
        c1, c2, c3, c4 = [int(v) for v in in_channels]
        self.ppm = PyramidPoolingModule(c4, fpn_channels)
        self.lateral_convs = nn.ModuleList(
            [
                nn.Conv2d(c1, fpn_channels, kernel_size=1, bias=False),
                nn.Conv2d(c2, fpn_channels, kernel_size=1, bias=False),
                nn.Conv2d(c3, fpn_channels, kernel_size=1, bias=False),
            ]
        )
        self.fpn_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1, bias=False),
                    _group_norm_2d(fpn_channels),
                    nn.GELU(),
                )
                for _ in range(3)
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(fpn_channels * 4, fpn_channels, kernel_size=3, padding=1, bias=False),
            _group_norm_2d(fpn_channels),
            nn.GELU(),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(fpn_channels, out_channels, kernel_size=1),
        )

    def forward_features(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        c1, c2, c3, c4 = features
        p4 = self.ppm(c4)
        laterals = [self.lateral_convs[0](c1), self.lateral_convs[1](c2), self.lateral_convs[2](c3), p4]
        for idx in range(2, -1, -1):
            target_size = laterals[idx].shape[-2:]
            laterals[idx] = laterals[idx] + F.interpolate(
                laterals[idx + 1], size=target_size, mode='bilinear', align_corners=False
            )
            laterals[idx] = self.fpn_convs[idx](laterals[idx])

        target_hw = laterals[0].shape[-2:]
        fused = torch.cat(
            [
                laterals[0],
                F.interpolate(laterals[1], size=target_hw, mode='bilinear', align_corners=False),
                F.interpolate(laterals[2], size=target_hw, mode='bilinear', align_corners=False),
                F.interpolate(laterals[3], size=target_hw, mode='bilinear', align_corners=False),
            ],
            dim=1,
        )
        return self.fuse[:-1](fused)

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        fused = self.forward_features(features)
        return self.fuse[-1](fused)


class SwinUPerNetBinary(nn.Module):
    """Real Swin Transformer backbone + UPerNet decoder for binary segmentation."""

    def __init__(
        self,
        input_channels: int = 3,
        *,
        backbone_name: str = 'swin_base_patch4_window7_224',
        freeze_backbone: bool = False,
        use_gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = _build_native_swin_backbone(backbone_name, int(input_channels))
        _enable_grad_checkpointing_if_supported(self.backbone, bool(use_gradient_checkpointing))
        _freeze_module(self.backbone, bool(freeze_backbone))
        feature_info = getattr(self.backbone, 'feature_info', None)
        if feature_info is None:
            raise RuntimeError('Backbone does not expose feature_info required by UPerNet head.')
        self._feature_channels = list(feature_info.channels())
        self.decode_head = UPerNetHead(in_channels=self._feature_channels, fpn_channels=256, out_channels=1)
        self.confidence_head = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        input_hw = x.shape[-2:]
        features = _normalize_backbone_features(
            _run_backbone_with_adaptive_input(self.backbone, x),
            self._feature_channels,
        )
        decoded = self.decode_head.forward_features(features)
        logits = self.decode_head.fuse[-1](decoded)
        logits = F.interpolate(logits, size=input_hw, mode='bilinear', align_corners=False)
        confidence = self.confidence_head(decoded)
        confidence = F.interpolate(confidence, size=input_hw, mode='bilinear', align_corners=False)
        return {'mask': logits, 'confidence': confidence}


class PixelDecoderFPN(nn.Module):
    """Lightweight FPN-like pixel decoder for Mask2Former-style masks."""

    def __init__(self, in_channels: Sequence[int], embed_dim: int = 256) -> None:
        super().__init__()
        c1, c2, c3, c4 = [int(v) for v in in_channels]
        self.proj4 = nn.Conv2d(c4, embed_dim, kernel_size=1, bias=False)
        self.proj3 = nn.Conv2d(c3, embed_dim, kernel_size=1, bias=False)
        self.proj2 = nn.Conv2d(c2, embed_dim, kernel_size=1, bias=False)
        self.proj1 = nn.Conv2d(c1, embed_dim, kernel_size=1, bias=False)
        self.conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            _group_norm_2d(embed_dim),
            nn.GELU(),
        )

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        c1, c2, c3, c4 = features
        x = self.proj4(c4)
        x = F.interpolate(x, size=c3.shape[-2:], mode='bilinear', align_corners=False) + self.proj3(c3)
        x = self.conv(x)
        x = F.interpolate(x, size=c2.shape[-2:], mode='bilinear', align_corners=False) + self.proj2(c2)
        x = self.conv(x)
        x = F.interpolate(x, size=c1.shape[-2:], mode='bilinear', align_corners=False) + self.proj1(c1)
        x = self.conv(x)
        return x


class Mask2FormerLikeHead(nn.Module):
    """Simplified Mask2Former head producing binary logits."""

    def __init__(self, in_channels: Sequence[int], embed_dim: int = 256, num_queries: int = 100) -> None:
        super().__init__()
        self.pixel_decoder = PixelDecoderFPN(in_channels=in_channels, embed_dim=embed_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation='gelu',
        )
        self.query_embed = nn.Embedding(num_queries, embed_dim)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=3)
        self.class_embed = nn.Linear(embed_dim, 2)  # foreground / no-object
        self.mask_embed = nn.Linear(embed_dim, embed_dim)

    def forward_features(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        return self.pixel_decoder(features)

    def decode_logits(self, mask_features: torch.Tensor) -> torch.Tensor:
        b, c, h, w = mask_features.shape
        memory = mask_features.flatten(2).transpose(1, 2)  # [B, HW, C]
        query = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)  # [B, Q, C]
        decoded = self.transformer_decoder(tgt=query, memory=memory)  # [B, Q, C]
        class_logits = self.class_embed(decoded)  # [B, Q, 2]
        mask_embed = self.mask_embed(decoded)  # [B, Q, C]
        mask_logits = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_features)  # [B, Q, H, W]

        class_prob = torch.softmax(class_logits, dim=-1)[..., 0]  # foreground score per query
        fused = torch.einsum('bq,bqhw->bhw', class_prob, mask_logits).unsqueeze(1)  # [B,1,H,W]
        return fused

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        return self.decode_logits(self.forward_features(features))


class Mask2FormerSwinBinary(nn.Module):
    """Real Swin Transformer backbone + lightweight Mask2Former-style decoder."""

    def __init__(
        self,
        input_channels: int = 3,
        *,
        backbone_name: str = 'swin_base_patch4_window7_224',
        freeze_backbone: bool = False,
        use_gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = _build_native_swin_backbone(backbone_name, int(input_channels))
        _enable_grad_checkpointing_if_supported(self.backbone, bool(use_gradient_checkpointing))
        _freeze_module(self.backbone, bool(freeze_backbone))
        feature_info = getattr(self.backbone, 'feature_info', None)
        if feature_info is None:
            raise RuntimeError('Backbone does not expose feature_info required by Mask2Former head.')
        self._feature_channels = list(feature_info.channels())
        self.decode_head = Mask2FormerLikeHead(in_channels=self._feature_channels, embed_dim=256, num_queries=100)
        self.confidence_head = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        input_hw = x.shape[-2:]
        features = _normalize_backbone_features(
            _run_backbone_with_adaptive_input(self.backbone, x),
            self._feature_channels,
        )
        decoded = self.decode_head.forward_features(features)
        logits = self.decode_head.decode_logits(decoded)
        logits = F.interpolate(logits, size=input_hw, mode='bilinear', align_corners=False)
        confidence = self.confidence_head(decoded)
        confidence = F.interpolate(confidence, size=input_hw, mode='bilinear', align_corners=False)
        return {'mask': logits, 'confidence': confidence}


@register_model(name='Swin UPerNet B', model_type=ModelType.experimental)
class SwinUPerNetB(SwinUPerNetBinary):
    """Swin-Base + UPerNet (binary segmentation logits)."""

    def __init__(self, input_channels: int = 3, **kwargs) -> None:
        kwargs.setdefault('backbone_name', 'swin_base_patch4_window7_224')
        kwargs.setdefault('freeze_backbone', _env_flag('NEURALIMAGE_FREEZE_BACKBONE', False))
        kwargs.setdefault('use_gradient_checkpointing', _env_flag('NEURALIMAGE_SWIN_GRAD_CHECKPOINTING', True))
        super().__init__(input_channels=input_channels, **kwargs)


@register_model(name='Swin UPerNet L', model_type=ModelType.experimental)
class SwinUPerNetL(SwinUPerNetBinary):
    """Swin-Large + UPerNet (binary segmentation logits)."""

    def __init__(self, input_channels: int = 3, **kwargs) -> None:
        kwargs.setdefault('backbone_name', 'swin_large_patch4_window7_224')
        kwargs.setdefault('freeze_backbone', _env_flag('NEURALIMAGE_FREEZE_BACKBONE', False))
        kwargs.setdefault('use_gradient_checkpointing', _env_flag('NEURALIMAGE_SWIN_GRAD_CHECKPOINTING', True))
        super().__init__(input_channels=input_channels, **kwargs)


@register_model(name='Mask2Former Swin B', model_type=ModelType.experimental)
class Mask2FormerSwinB(Mask2FormerSwinBinary):
    """Swin-Base + Mask2Former-like decoder (binary segmentation logits)."""

    def __init__(self, input_channels: int = 3, **kwargs) -> None:
        kwargs.setdefault('backbone_name', 'swin_base_patch4_window7_224')
        kwargs.setdefault('freeze_backbone', _env_flag('NEURALIMAGE_FREEZE_BACKBONE', False))
        kwargs.setdefault('use_gradient_checkpointing', _env_flag('NEURALIMAGE_SWIN_GRAD_CHECKPOINTING', True))
        super().__init__(input_channels=input_channels, **kwargs)


@register_model(name='Mask2Former Swin L', model_type=ModelType.experimental)
class Mask2FormerSwinL(Mask2FormerSwinBinary):
    """Swin-Large + Mask2Former-like decoder (binary segmentation logits)."""

    def __init__(self, input_channels: int = 3, **kwargs) -> None:
        kwargs.setdefault('backbone_name', 'swin_large_patch4_window7_224')
        kwargs.setdefault('freeze_backbone', _env_flag('NEURALIMAGE_FREEZE_BACKBONE', False))
        kwargs.setdefault('use_gradient_checkpointing', _env_flag('NEURALIMAGE_SWIN_GRAD_CHECKPOINTING', True))
        super().__init__(input_channels=input_channels, **kwargs)
