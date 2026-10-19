"""Restore a Kraken inference parameter export into the existing desktop state."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import math

from neuralimage.application.dto import SettingsState


def restore_inference_settings(payload: Mapping[str, object], current: SettingsState) -> SettingsState:
    if payload.get("format") != "neuralimage-inference-settings" or payload.get("version") != 1:
        raise ValueError("Неподдерживаемый формат параметров бинаризации")
    patch = payload.get("patch_size")
    if not isinstance(patch, (list, tuple)) or len(patch) != 2:
        raise ValueError("patch_size должен содержать два размера")
    if any(isinstance(v, bool) or not isinstance(v, int) or not 16 <= v <= 8192 for v in patch):
        raise ValueError("Размер патча должен быть от 16 до 8192")
    batch, overlap, kernel = (payload.get(k) for k in ("batch_size", "overlap", "postprocess_kernel_size"))
    for name, value, low, high in (
        ("batch_size", batch, 1, 1024),
        ("overlap", overlap, 0, min(patch) - 1),
        ("postprocess_kernel_size", kernel, 1, 255),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f"Недопустимое значение {name}")
    threshold = payload.get("threshold")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError("Порог бинаризации должен быть от 0 до 1")
    for name in ("use_auto_threshold", "tta", "postprocess_enabled"):
        if not isinstance(payload.get(name), bool):
            raise ValueError(f"{name} должен иметь значение true или false")
    confidence_tta = payload.get("confidence_tta", False)
    if not isinstance(confidence_tta, bool):
        raise ValueError("confidence_tta должен иметь значение true или false")
    confidence_mode = payload.get("confidence_save_mode", "off")
    if not isinstance(confidence_mode, str) or confidence_mode not in {"off", "separate_grayscale"}:
        raise ValueError("Неизвестный режим confidence")
    return replace(
        current,
        recognition_patch_size=tuple(patch),
        recognition_batch_size=batch,
        sync_patch_sizes=False,
        patch_batch_sync_mode="off",
        overlap=overlap,
        recognition_binarize_output=True,
        recognition_use_auto_threshold=payload["use_auto_threshold"],
        recognition_threshold=float(threshold),
        recognition_tta_enabled=payload["tta"],
        recognition_postprocess=payload["postprocess_enabled"],
        recognition_postprocess_kernel_size=kernel,
        confidence_save_mode=confidence_mode,
        confidence_tta_enabled=confidence_tta,
    )
