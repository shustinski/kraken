from __future__ import annotations

import collections
import os
import re
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from lib.file_retry import retry_file_read
import model

from . import CNN_Models, blocks, dual_scale_models, transformer_segmentation
from .registrator import create_model, get_registered_models


MODEL_ARTIFACT_FORMAT = 'neuralimage_model_artifact'
MODEL_ARTIFACT_VERSION = 2
_UNSAFE_MODEL_LOAD_ENV = 'NEURALIMAGE_ALLOW_UNSAFE_MODEL_LOAD'
_TRUE_VALUES = {'1', 'true', 'yes', 'on'}
_MAX_DYNAMIC_SAFE_GLOBAL_RETRIES = 32
_UNSUPPORTED_GLOBAL_PATTERN = re.compile(r'Unsupported global: GLOBAL ([\w\.]+)')
_DYNAMIC_SAFE_GLOBAL_PREFIXES = (
    'model.NeuralNetwork.',
    'torch.nn.',
    'collections.',
)
_SAFE_GLOBAL_ROOTS: dict[str, Any] = {
    'collections': collections,
    'model': model,
    'torch': torch,
}


def _iter_internal_module_classes(module: Any) -> list[type[Any]]:
    classes: list[type[Any]] = []
    for value in vars(module).values():
        if not isinstance(value, type):
            continue
        if not issubclass(value, nn.Module):
            continue
        module_name = str(getattr(value, '__module__', ''))
        if module_name.startswith('model.NeuralNetwork.'):
            classes.append(value)
    return classes


def _iter_torch_nn_classes() -> list[type[Any]]:
    classes: list[type[Any]] = []
    for value in vars(nn).values():
        if not isinstance(value, type):
            continue
        module_name = str(getattr(value, '__module__', ''))
        if module_name.startswith('torch.nn.modules.') or module_name == 'torch.nn.parameter':
            classes.append(value)
    return classes


def _resolve_safe_pickle_globals() -> list[Any]:
    candidates: list[Any] = []
    candidates.extend(_iter_torch_nn_classes())
    candidates.extend(get_registered_models().values())
    candidates.extend(_iter_internal_module_classes(CNN_Models))
    candidates.extend(_iter_internal_module_classes(dual_scale_models))
    candidates.extend(_iter_internal_module_classes(transformer_segmentation))
    candidates.extend(_iter_internal_module_classes(blocks))

    seen: set[tuple[str, str]] = set()
    resolved: list[Any] = []
    for item in candidates:
        key = (str(getattr(item, '__module__', '')), str(getattr(item, '__qualname__', '')))
        if key in seen:
            continue
        seen.add(key)
        resolved.append(item)
    return resolved


def _is_dynamic_safe_global_path(path: str) -> bool:
    return any(path.startswith(prefix) for prefix in _DYNAMIC_SAFE_GLOBAL_PREFIXES)


def _extract_unsupported_global_path(error: Exception) -> str | None:
    match = _UNSUPPORTED_GLOBAL_PATTERN.search(str(error))
    if match is None:
        return None
    return str(match.group(1))


def _resolve_global_symbol(path: str) -> Any | None:
    parts = [part for part in str(path).split('.') if part]
    if len(parts) < 2:
        return None
    target = _SAFE_GLOBAL_ROOTS.get(parts[0])
    if target is None:
        return None
    try:
        for attr in parts[1:]:
            target = getattr(target, attr)
    except Exception:
        return None
    return target


def _torch_load_with_safe_globals(
    path: Path,
    *,
    map_location: str | torch.device | None,
    allowed_globals: list[Any],
) -> Any:
    serialization = getattr(torch, 'serialization', None)
    safe_globals_ctx = getattr(serialization, 'safe_globals', None) if serialization is not None else None
    add_safe_globals = getattr(serialization, 'add_safe_globals', None) if serialization is not None else None
    if callable(safe_globals_ctx) and allowed_globals:
        with safe_globals_ctx(allowed_globals):
            return retry_file_read(
                lambda: torch.load(path, map_location=map_location, weights_only=True),
                path=path,
            )

    if callable(add_safe_globals) and allowed_globals:
        add_safe_globals(allowed_globals)
        return retry_file_read(
            lambda: torch.load(path, map_location=map_location, weights_only=True),
            path=path,
        )

    with nullcontext():
        return retry_file_read(
            lambda: torch.load(path, map_location=map_location, weights_only=True),
            path=path,
        )


def _load_torch_safe_weights_only(
    path: Path,
    *,
    map_location: str | torch.device | None,
) -> Any:
    allowed_globals = _resolve_safe_pickle_globals()
    added_symbols: set[str] = set()
    for _ in range(_MAX_DYNAMIC_SAFE_GLOBAL_RETRIES):
        try:
            return _torch_load_with_safe_globals(
                path,
                map_location=map_location,
                allowed_globals=allowed_globals,
            )
        except Exception as error:
            global_path = _extract_unsupported_global_path(error)
            if not global_path:
                raise
            if global_path in added_symbols:
                raise
            if not _is_dynamic_safe_global_path(global_path):
                raise
            symbol = _resolve_global_symbol(global_path)
            if symbol is None:
                raise
            allowed_globals.append(symbol)
            added_symbols.add(global_path)
    raise RuntimeError('Unable to resolve safe globals for weights-only load.')


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


def _coerce_model_kwargs(raw_kwargs: Any) -> dict[str, Any]:
    if not isinstance(raw_kwargs, Mapping):
        return {}

    normalized: dict[str, Any] = {}
    for key, value in dict(raw_kwargs).items():
        if not isinstance(key, str):
            continue
        if isinstance(value, (bool, int, float, str)):
            normalized[key] = value
            continue
        if isinstance(value, (list, tuple)) and all(isinstance(item, (bool, int, float, str)) for item in value):
            normalized[key] = tuple(value)
    return normalized


def _coerce_artifact_metadata(raw_metadata: Any) -> dict[str, Any]:
    if not isinstance(raw_metadata, Mapping):
        return {}

    normalized: dict[str, Any] = {}
    for key, value in dict(raw_metadata).items():
        if not isinstance(key, str):
            continue
        if isinstance(value, (bool, int, float, str)):
            normalized[key] = value
            continue
        if isinstance(value, Mapping):
            nested = _coerce_artifact_metadata(value)
            if nested:
                normalized[key] = nested
            continue
        if isinstance(value, (list, tuple)) and all(isinstance(item, (bool, int, float, str)) for item in value):
            normalized[key] = tuple(value)
    return normalized


def _attach_model_metadata(
    model: nn.Module,
    *,
    model_name: str,
    input_channels: int,
    model_kwargs: Mapping[str, Any] | None = None,
    artifact_metadata: Mapping[str, Any] | None = None,
) -> nn.Module:
    setattr(model, '_neuralimage_model_name', str(model_name))
    setattr(model, '_neuralimage_input_channels', int(input_channels))
    setattr(model, '_neuralimage_model_kwargs', _coerce_model_kwargs(model_kwargs))
    setattr(model, '_neuralimage_artifact_metadata', _coerce_artifact_metadata(artifact_metadata))
    return model


def _build_model_from_state_dict(
    state_dict: Mapping[str, Any],
    *,
    model_name: str | None,
    input_channels: int | None,
    model_kwargs: Mapping[str, Any] | None = None,
    artifact_metadata: Mapping[str, Any] | None = None,
) -> nn.Module | None:
    if not model_name:
        return None

    resolved_channels = int(input_channels) if input_channels is not None else _infer_channels_from_state_dict(state_dict)
    if resolved_channels is None:
        return None

    normalized_model_kwargs = _coerce_model_kwargs(model_kwargs)
    model = create_model(str(model_name), int(resolved_channels), **normalized_model_kwargs)
    load_result = model.load_state_dict(state_dict, strict=False)
    allowed_missing_prefixes = ('confidence_head.',)
    missing_keys = [
        str(key)
        for key in getattr(load_result, 'missing_keys', [])
        if not any(str(key).startswith(prefix) for prefix in allowed_missing_prefixes)
    ]
    unexpected_keys = [str(key) for key in getattr(load_result, 'unexpected_keys', [])]
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            'Model state_dict is incompatible with the requested architecture. '
            f'missing_keys={missing_keys}, unexpected_keys={unexpected_keys}'
        )
    return _attach_model_metadata(
        model,
        model_name=str(model_name),
        input_channels=int(resolved_channels),
        model_kwargs=normalized_model_kwargs,
        artifact_metadata=artifact_metadata,
    )


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
                model_kwargs=payload_map.get('model_kwargs'),
                artifact_metadata=payload_map.get('metadata'),
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
                model_kwargs=payload_map.get('model_kwargs'),
                artifact_metadata=payload_map.get('metadata'),
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
    model_kwargs: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    normalized_model_kwargs = _coerce_model_kwargs(
        model_kwargs if model_kwargs is not None else getattr(model, '_neuralimage_model_kwargs', None)
    )
    normalized_metadata = _coerce_artifact_metadata(
        metadata if metadata is not None else getattr(model, '_neuralimage_artifact_metadata', None)
    )
    payload = {
        'format': MODEL_ARTIFACT_FORMAT,
        'version': MODEL_ARTIFACT_VERSION,
        'model_name': str(model_name),
        'input_channels': int(input_channels),
        'model_kwargs': normalized_model_kwargs,
        'metadata': normalized_metadata,
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
        payload = _load_torch_safe_weights_only(path, map_location=map_location)
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

    legacy_payload = retry_file_read(
        lambda: torch.load(path, map_location=map_location, weights_only=False),
        path=path,
    )
    legacy_model = _load_model_from_safe_payload(
        legacy_payload,
        model_name_fallback=model_name_fallback,
        input_channels_fallback=input_channels_fallback,
    )
    if legacy_model is not None:
        return legacy_model
    raise TypeError(f'Unsupported legacy model payload type: {type(legacy_payload)!r}')
