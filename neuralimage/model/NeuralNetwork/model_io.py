from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .registrator import create_model


MODEL_ARTIFACT_FORMAT = 'neuralimage_model_artifact'
MODEL_ARTIFACT_VERSION = 1
_UNSAFE_MODEL_LOAD_ENV = 'NEURALIMAGE_ALLOW_UNSAFE_MODEL_LOAD'
_TRUE_VALUES = {'1', 'true', 'yes', 'on'}


def _resolve_allow_unsafe_model_load(explicit: bool | None = None) -> bool:
    if explicit is not None:
        return bool(explicit)
    raw = str(os.getenv(_UNSAFE_MODEL_LOAD_ENV, '0')).strip().lower()
    return raw in _TRUE_VALUES


def _infer_channels_from_state_dict(state_dict: Mapping[str, Any]) -> int | None:
    for tensor in state_dict.values():
        if isinstance(tensor, torch.Tensor) and tensor.ndim >= 2:
            return int(tensor.shape[1])
    return None


def _attach_model_metadata(model: nn.Module, *, model_name: str, input_channels: int) -> nn.Module:
    setattr(model, '_neuralimage_model_name', str(model_name))
    setattr(model, '_neuralimage_input_channels', int(input_channels))
    return model


def _build_model_from_state_dict(
    state_dict: Mapping[str, Any],
    *,
    model_name: str | None,
    input_channels: int | None,
) -> nn.Module | None:
    if not model_name:
        return None

    resolved_channels = int(input_channels) if input_channels is not None else _infer_channels_from_state_dict(state_dict)
    if resolved_channels is None:
        return None

    model = create_model(str(model_name), int(resolved_channels))
    model.load_state_dict(state_dict)
    return _attach_model_metadata(model, model_name=str(model_name), input_channels=int(resolved_channels))


def _load_model_from_safe_payload(
    payload: Any,
    *,
    model_name_fallback: str | None,
    input_channels_fallback: int | None,
) -> nn.Module | None:
    if isinstance(payload, nn.Module):
        return payload

    if isinstance(payload, Mapping):
        payload_map = dict(payload)
        if payload_map.get('format') == MODEL_ARTIFACT_FORMAT:
            state_dict = payload_map.get('state_dict')
            if not isinstance(state_dict, Mapping):
                raise TypeError('Model artifact is missing a valid "state_dict".')
            return _build_model_from_state_dict(
                state_dict,
                model_name=str(payload_map.get('model_name') or model_name_fallback or ''),
                input_channels=(
                    int(payload_map['input_channels'])
                    if payload_map.get('input_channels') is not None
                    else input_channels_fallback
                ),
            )

        state_dict = payload_map.get('model_state_dict')
        if isinstance(state_dict, Mapping):
            return _build_model_from_state_dict(
                state_dict,
                model_name=str(payload_map.get('model_name') or model_name_fallback or ''),
                input_channels=(
                    int(payload_map['input_channels'])
                    if payload_map.get('input_channels') is not None
                    else input_channels_fallback
                ),
            )

        if all(isinstance(value, torch.Tensor) for value in payload_map.values()):
            return _build_model_from_state_dict(
                payload_map,
                model_name=model_name_fallback,
                input_channels=input_channels_fallback,
            )

    return None


def save_model_artifact(
    model: nn.Module,
    save_path: str | Path,
    *,
    model_name: str,
    input_channels: int,
) -> None:
    payload = {
        'format': MODEL_ARTIFACT_FORMAT,
        'version': MODEL_ARTIFACT_VERSION,
        'model_name': str(model_name),
        'input_channels': int(input_channels),
        'state_dict': model.state_dict(),
    }
    torch.save(payload, save_path)


def load_model_artifact(
    model_source: str | Path,
    *,
    map_location: str | torch.device | None = 'cpu',
    model_name_fallback: str | None = None,
    input_channels_fallback: int | None = None,
    allow_unsafe_legacy_pickle: bool | None = None,
) -> nn.Module:
    path = Path(model_source)

    safe_load_error: Exception | None = None
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
        model = _load_model_from_safe_payload(
            payload,
            model_name_fallback=model_name_fallback,
            input_channels_fallback=input_channels_fallback,
        )
        if model is not None:
            return model
    except Exception as error:
        safe_load_error = error

    if not _resolve_allow_unsafe_model_load(allow_unsafe_legacy_pickle):
        extra = f' Safe load error: {safe_load_error}' if safe_load_error is not None else ''
        raise RuntimeError(
            'Model file is not in NeuralImage safe format or requires unsafe pickle loading. '
            f'Set {_UNSAFE_MODEL_LOAD_ENV}=1 only for trusted legacy models.{extra}'
        )

    legacy_payload = torch.load(path, map_location=map_location, weights_only=False)
    legacy_model = _load_model_from_safe_payload(
        legacy_payload,
        model_name_fallback=model_name_fallback,
        input_channels_fallback=input_channels_fallback,
    )
    if legacy_model is not None:
        return legacy_model
    raise TypeError(f'Unsupported legacy model payload type: {type(legacy_payload)!r}')
