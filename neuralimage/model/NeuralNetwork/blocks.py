from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def extract_mask_outputs(outputs: Any) -> Any:
    if isinstance(outputs, dict):
        if 'mask' not in outputs:
            raise KeyError('Structured model outputs must contain a "mask" entry.')
        return outputs['mask']
    return outputs


def extract_confidence_output(outputs: Any) -> torch.Tensor | None:
    if not isinstance(outputs, dict):
        return None
    confidence = outputs.get('confidence')
    if confidence is None:
        return None
    if not torch.is_tensor(confidence):
        raise TypeError('Structured model confidence output must be a tensor.')
    return confidence


def resolve_group_norm_groups(channels: int, max_groups: int = 8) -> int:
    resolved_channels = max(1, int(channels))
    resolved_max_groups = max(1, min(int(max_groups), resolved_channels))
    for groups in range(resolved_max_groups, 0, -1):
        if resolved_channels % groups == 0:
            return groups
    return 1


def build_norm_2d(
    num_channels: int,
    *,
    norm: str = 'bn',
    gn_groups: int = 8,
) -> nn.Module:
    normalized = str(norm or 'bn').strip().lower()
    if normalized == 'bn':
        return nn.BatchNorm2d(num_channels)
    if normalized == 'gn':
        return nn.GroupNorm(resolve_group_norm_groups(num_channels, max_groups=gn_groups), num_channels)
    return nn.InstanceNorm2d(num_channels, affine=True)


def build_activation(act: str = 'gelu') -> nn.Module:
    normalized = str(act or 'gelu').strip().lower()
    if normalized == 'gelu':
        return nn.GELU()
    if normalized == 'silu':
        return nn.SiLU()
    return nn.LeakyReLU(0.2, inplace=True)


def build_conv_sequence(
    layer_specs: Sequence[tuple[int, int, float, bool]],
    *,
    norm: str = 'bn',
    act: str = 'gelu',
    gn_groups: int = 8,
) -> nn.Sequential:
    return nn.Sequential(
        *[
            ConvBlock(
                in_ch,
                out_ch,
                norm=norm,
                act=act,
                droput=dropout,
                pooling=pooling,
                gn_groups=gn_groups,
            )
            for in_ch, out_ch, dropout, pooling in layer_specs
        ]
    )


class DeepSupervisionMixin:
    def _init_deep_supervision(self, deep_supervision=False):
        self.deep_supervision = bool(deep_supervision)

    def _merge_deep_supervision_outputs(self, primary, auxiliary_outputs):
        if (not getattr(self, 'deep_supervision', False)) or (not self.training):
            return primary

        merged = [primary]
        target_size = primary.shape[-2:]
        for aux_output in auxiliary_outputs:
            if aux_output is None:
                continue
            if aux_output.shape[-2:] != target_size:
                aux_output = F.interpolate(aux_output, size=target_size, mode='bilinear', align_corners=False)
            merged.append(aux_output)
        return tuple(merged)

    def _build_model_outputs(self, primary, auxiliary_outputs=(), confidence=None):
        mask_outputs = self._merge_deep_supervision_outputs(primary, auxiliary_outputs)
        if confidence is None:
            return mask_outputs
        target_size = primary.shape[-2:]
        if confidence.shape[-2:] != target_size:
            confidence = F.interpolate(confidence, size=target_size, mode='bilinear', align_corners=False)
        return {
            'mask': mask_outputs,
            'confidence': confidence,
        }


class SegmentationHeadBundle(nn.Module):
    def __init__(
        self,
        primary_channels: int,
        *,
        aux_channels: Sequence[int] = (),
        primary_kernel_size: int = 1,
        confidence_kernel_size: int | None = None,
        aux_kernel_size: int = 1,
    ) -> None:
        super().__init__()
        confidence_kernel = int(confidence_kernel_size or primary_kernel_size)
        self.primary = nn.Conv2d(
            primary_channels,
            1,
            kernel_size=int(primary_kernel_size),
            padding=int(primary_kernel_size) // 2,
        )
        self.confidence = nn.Conv2d(
            primary_channels,
            1,
            kernel_size=confidence_kernel,
            padding=confidence_kernel // 2,
        )
        self.auxiliary = nn.ModuleList(
            nn.Conv2d(
                int(channels),
                1,
                kernel_size=int(aux_kernel_size),
                padding=int(aux_kernel_size) // 2,
            )
            for channels in aux_channels
        )

    def forward(
        self,
        primary_features: torch.Tensor,
        auxiliary_features: Sequence[torch.Tensor] = (),
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        primary = self.primary(primary_features)
        confidence = self.confidence(primary_features)
        auxiliary_outputs = tuple(
            head(feature)
            for head, feature in zip(self.auxiliary, auxiliary_features)
        )
        return primary, confidence, auxiliary_outputs

class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch,k=3, s=1,p=1,gn_groups=8):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.norm = build_norm_2d(out_ch, norm='gn', gn_groups=gn_groups)
        self.act = build_activation('silu')

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.norm(x)
        return self.act(x)

class ResDSBlock(nn.Module):
    """Two DSConv with residual (when shape matches)."""
    def __init__(self, in_ch, out_ch, gn_groups=8, dropout: float = 0.0):
        super().__init__()
        self.conv1 = DSConv(in_ch, out_ch, gn_groups=gn_groups)
        self.conv2 = DSConv(out_ch, out_ch, gn_groups=gn_groups)
        self.use_dropout = dropout > 0.0
        self.dropout = nn.Dropout2d(p=dropout) if self.use_dropout else nn.Identity()
        self.skip = None
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout(x)
        if self.skip is not None:
            identity = self.skip(identity)
        return x + identity


class Down(nn.Module):
    """Downsample by stride-2 depthwise-separable conv."""
    def __init__(self, in_ch, out_ch, gn_groups=8, dropout: float = 0.0):
        super().__init__()
        self.down = DSConv(in_ch, out_ch, s=2, gn_groups=gn_groups)   # stride=2
        self.block = ResDSBlock(out_ch, out_ch, gn_groups=gn_groups, dropout=dropout)

    def forward(self, x):
        x = self.down(x)
        return self.block(x)


class Up(nn.Module):
    """Upsample + concat skip + refine."""
    def __init__(self, in_ch, skip_ch, out_ch, gn_groups=8, dropout: float = 0.0):
        super().__init__()
        self.refine = ResDSBlock(in_ch + skip_ch, out_ch, gn_groups=gn_groups, dropout=dropout)

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        # spatial safety (in case H/W not divisible by 16)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.refine(x)

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch,
                 norm='bn',
                 act='gelu',
                 droput:float=0,
                 stride=1,
                 pooling:bool=False,
                 gn_groups: int = 8):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm = build_norm_2d(out_ch, norm=norm, gn_groups=gn_groups)
        self.act = build_activation(act)

        self.use_dropout = bool(droput)
        self.dropout = nn.Dropout2d(p=droput)

        self.pooling = bool(pooling)
        if pooling:
            self.pooling_layer = nn.MaxPool2d(2, stride=2, padding=0)
        else:
            self.pooling_layer = None


    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        if self.use_dropout:
            x = self.dropout(x)
        if self.pooling and self.pooling_layer is not None:
            x = self.pooling_layer(x)
        return x

class ConvTransposeEncoder(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels,
                 norm='bn',
                 act='gelu',
                 droput: float = 0,
                 stride=1,
                 gn_groups: int = 8,
                 ):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3,
                                                 stride=2, padding=1, output_padding=1)
        self.conv1 = ConvBlock(
            out_channels + skip_channels,
            out_channels,
            norm=norm,
            act=act,
            droput=droput,
            pooling=False,
            gn_groups=gn_groups,
        )
        self.conv2 = ConvBlock(
            out_channels,
            out_channels,
            norm=norm,
            act=act,
            droput=droput,
            pooling=False,
            gn_groups=gn_groups,
        )

    def forward(self, x, skip=None):
        # Transpose convolution
        x = self.conv_transpose(x)
        if skip is not None:
            skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)

        return x

class ConvUpsampleEncoder(nn.Module):
    def __init__(self, skip_channels, out_channels,
                 norm='bn',
                 act='gelu',
                 droput: float = 0,
                 stride=1,
                 gn_groups: int = 8,
                 ):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = ConvBlock(
            out_channels + skip_channels,
            out_channels,
            norm=norm,
            act=act,
            droput=droput,
            pooling=False,
            gn_groups=gn_groups,
        )
        self.conv2 = ConvBlock(
            out_channels,
            out_channels,
            norm=norm,
            act=act,
            droput=droput,
            pooling=False,
            gn_groups=gn_groups,
        )

    def forward(self, x, skip=None):
        # Transpose convolution
        x = self.upsample(x)
        if skip is not None:
            skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)

        return x

class Bottleneck(nn.Module):

    def __init__(self, in_ch: int, *, norm: str = 'bn', act: str = 'leaky', dropout: float = 0.3, gn_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(in_ch, in_ch*2, norm=norm, act=act, droput=dropout, pooling=False, gn_groups=gn_groups),
            ConvBlock(in_ch*2, in_ch*2, norm=norm, act=act, droput=dropout, pooling=False, gn_groups=gn_groups),
        )

    def forward(self, x):
        x = self.block(x)
        return x

class SEBlock(nn.Module):
    def __init__(self, ch, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.extinstion = nn.Sequential(
            nn.Linear(ch, ch//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch//reduction, ch, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs,c,_,_ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.extinstion(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)



class ResBlock(nn.Module):
    def __init__(self, ch, norm='gn', act='gelu', use_se=False):
        super().__init__()
        self.conv1 = ConvBlock(ch, ch, norm=norm, act=act)
        self.conv2 = ConvBlock(ch, ch, norm=norm, act=act)
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(ch)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.use_se:
            x = self.se(x)
        return x + residual



class UpBlock(nn.Module):
    """
    Upsample (ConvTranspose2d) → concat со skip‑тензором →
    ConvBlock (внутри учитывает реальное количество каналов).
    """
    def __init__(self,
                 in_ch,          # каналы входа в ConvTranspose2d
                 out_ch,         # каналы после up‑sampling
                 skip_ch=None,   # каналы у skip‑тензора (по умолчанию = out_ch)
                 use_se=False,
                 norm='gn',
                 act='gelu'):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch,
                                     kernel_size=2, stride=2)

        if skip_ch is None:
            skip_ch = 0
        conv_in = out_ch + skip_ch          # после конкатенации
        self.conv = ConvBlock(conv_in, out_ch,
                              norm=norm, act=act, pooling=False)
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(out_ch)

    def forward(self, x):
        x = self.up(x)                                            # ↑2
        x = self.conv(x)
        if self.use_se:
            x = self.se(x)
        return x
