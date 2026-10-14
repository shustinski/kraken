"""recognition_pipeline: load preview array."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any
import numpy as np
import torch
from kraken_core.asset_source import AssetSource
from neuralimage.lib.images import open_image

def _load_preview_array(path: Path | str | None, asset_source: AssetSource | None = None) -> np.ndarray | None:
    if path is None:
        return None
    try:
        with open_image(path, asset_source) as source_image:
            if source_image.mode == 'L':
                preview_image = source_image.copy()
            elif source_image.mode == 'RGB':
                preview_image = source_image.copy()
            else:
                preview_image = source_image.convert('RGB')
    except Exception:
        if asset_source is not None:
            raise
        return None

    preview_array = np.asarray(preview_image)
    if preview_array.ndim == 2:
        return np.ascontiguousarray(preview_array.astype(np.uint8, copy=False))
    if preview_array.ndim == 3 and preview_array.shape[2] >= 3:
        if preview_array.shape[2] != 3:
            preview_array = preview_array[:, :, :3]
        return np.ascontiguousarray(preview_array.astype(np.uint8, copy=False))
    return None

def _publish_recognition_preview(
    *,
    publish: Publisher,
    source_path: Path | str | None,
    output_path: Path | str | None,
    frame_name: str,
    asset_source: AssetSource | None = None,
) -> None:
    original_image = _load_preview_array(source_path, asset_source)
    output_image = _load_preview_array(output_path)
    if original_image is None and output_image is None:
        return
    publish(
        'metrics',
        {
            'type': 'recognition_preview',
            'sample_name': str(frame_name).strip(),
            'image': original_image,
            'outputs': output_image,
        },
    )

def _store_payload_array_for_multiprocessing(payload: dict[str, Any], key: str) -> None:
    value = payload.get(key)
    if not isinstance(value, np.ndarray):
        return
    contiguous = np.ascontiguousarray(value)
    if _should_use_torch_tensor_transport():
        payload[key] = torch.from_numpy(contiguous)
        return
    payload[key] = contiguous

def _restore_payload_array_from_multiprocessing(payload: dict[str, Any], key: str) -> dict[str, Any]:
    restored_payload = dict(payload)
    value = restored_payload.get(key)
    if torch.is_tensor(value):
        restored_payload[key] = value.numpy()
    return restored_payload

def _should_use_torch_tensor_transport() -> bool:
    return os.name != 'nt'
