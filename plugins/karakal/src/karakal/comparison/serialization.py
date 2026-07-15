"""Serialization helpers for versioned comparison sidecars."""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .models import FrameComparisonResult

SCHEMA_VERSION = 1


def comparison_result_to_json_dict(result: FrameComparisonResult) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "comparison": {
            "frame_id": result.frame_id,
            "mode": result.mode,
            "models": list(result.model_ids),
            "profile": result.profile,
            "parameters": {
                "threshold": result.cache_key.threshold,
                "consensus_threshold": result.cache_key.consensus_threshold,
                "connectivity": result.cache_key.connectivity,
                "pruning_min_length_px": result.cache_key.pruning_threshold,
                "algorithm_version": result.cache_key.algorithm_version,
                "evidence_provider_version": result.cache_key.evidence_provider_version,
            },
            "risk": dict(result.risk),
            "metrics": [
                {
                    "name": metric.name,
                    "value": metric.value,
                    "group": metric.group,
                    "description": metric.description,
                    "unit": metric.unit,
                    "valid": metric.valid,
                }
                for metric in result.metrics
            ],
            "events": [_safe_dataclass_dict(event) for event in result.events],
            "layers": {
                "raster": [
                    {
                        "layer_id": layer.layer_id,
                        "title": layer.title,
                        "opacity": layer.opacity,
                        "visible": layer.visible,
                        "layer_group": layer.layer_group,
                        "shape": tuple(int(value) for value in layer.image.shape),
                    }
                    for layer in result.raster_layers
                ],
                "vector": [_safe_dataclass_dict(layer) for layer in result.vector_layers],
            },
            "metadata": dict(result.metadata),
        },
    }


def write_comparison_sidecar(result: FrameComparisonResult, path: Path | str) -> None:
    Path(path).write_text(json.dumps(comparison_result_to_json_dict(result), indent=2, ensure_ascii=False), encoding="utf-8")


def _safe_dataclass_dict(value: object) -> dict[str, Any]:
    raw = asdict(value) if is_dataclass(value) else dict(value)  # type: ignore[arg-type]
    return _json_safe(raw)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return {"array_shape": tuple(int(item) for item in value.shape), "array_dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    return value
