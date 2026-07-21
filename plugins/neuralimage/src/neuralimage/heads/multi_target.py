from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn


DEFAULT_HEAD_OUTPUT_CHANNELS: dict[str, int] = {
    'boundary': 1,
    'skeleton': 1,
    'sdf': 1,
    'distance_transform': 1,
    'thickness': 1,
    'vertex': 1,
    'corner': 1,
    'endpoint': 1,
    'junction': 1,
    'orientation': 1,
    'tangent': 2,
    'curvature': 1,
    'topology': 1,
}


def resolve_head_output_channels(head_name: str) -> int:
    return int(DEFAULT_HEAD_OUTPUT_CHANNELS.get(str(head_name), 1))


class MultiTargetHeadBundle(nn.Module):
    """Extensible multi-head bundle for mask + auxiliary supervision heads."""

    def __init__(
        self,
        primary_channels: int,
        *,
        supervision_heads: Sequence[str] = (),
        aux_channels: Sequence[int] = (),
        primary_kernel_size: int = 1,
        confidence_kernel_size: int | None = None,
        aux_kernel_size: int = 1,
    ) -> None:
        super().__init__()
        confidence_kernel = int(confidence_kernel_size or primary_kernel_size)
        self.primary = nn.Conv2d(
            int(primary_channels),
            1,
            kernel_size=int(primary_kernel_size),
            padding=int(primary_kernel_size) // 2,
        )
        self.confidence = nn.Conv2d(
            int(primary_channels),
            1,
            kernel_size=confidence_kernel,
            padding=confidence_kernel // 2,
        )
        self.supervision_heads = nn.ModuleDict(
            {
                str(name): nn.Conv2d(
                    int(primary_channels),
                    resolve_head_output_channels(name),
                    kernel_size=int(primary_kernel_size),
                    padding=int(primary_kernel_size) // 2,
                )
                for name in supervision_heads
            }
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
        self._supervision_head_names = tuple(str(name) for name in supervision_heads)

    @property
    def supervision_head_names(self) -> tuple[str, ...]:
        return self._supervision_head_names

    def forward(
        self,
        primary_features: torch.Tensor,
        auxiliary_features: Sequence[torch.Tensor] = (),
        *,
        include_supervision: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...], dict[str, torch.Tensor]]:
        primary = self.primary(primary_features)
        confidence = self.confidence(primary_features)
        auxiliary_outputs = tuple(
            head(feature)
            for head, feature in zip(self.auxiliary, auxiliary_features)
        )
        supervision_outputs: dict[str, torch.Tensor] = {}
        if include_supervision:
            for name, head in self.supervision_heads.items():
                supervision_outputs[name] = head(primary_features)
        return primary, confidence, auxiliary_outputs, supervision_outputs


def build_training_output_dict(
    primary: torch.Tensor,
    *,
    confidence: torch.Tensor | None = None,
    auxiliary_outputs: Sequence[torch.Tensor] = (),
    supervision_outputs: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor | tuple[torch.Tensor, ...]]:
    mask_output: torch.Tensor | tuple[torch.Tensor, ...]
    if auxiliary_outputs:
        mask_output = (primary, *auxiliary_outputs)
    else:
        mask_output = primary
    payload: dict[str, torch.Tensor | tuple[torch.Tensor, ...]] = {'mask': mask_output}
    if confidence is not None:
        payload['confidence'] = confidence
    if supervision_outputs:
        payload.update(supervision_outputs)
    return payload


def extract_inference_mask(outputs: torch.Tensor | Mapping[str, torch.Tensor | tuple[torch.Tensor, ...]]) -> torch.Tensor:
    if isinstance(outputs, Mapping):
        mask = outputs.get('mask')
        if mask is None:
            raise KeyError('Structured outputs must contain "mask".')
        if isinstance(mask, (list, tuple)):
            return mask[0]
        return mask
    if isinstance(outputs, (list, tuple)):
        return outputs[0]
    return outputs
