"""repository: export ranked frames."""
from __future__ import annotations
import json
import shutil
from dataclasses import replace
from pathlib import Path
from neuralimage.Validation_gradient_widget_lite.core.backend_constants import (
    EXPORT_SELECTION_MODE_COUNT,
)
from neuralimage.Validation_gradient_widget_lite.core.domain import (
    BuildResult,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.analytics_worker_count import natural_sort_key

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.metric_higher_better import (
    compute_metric_percentiles,
    metric_value_for_record,
    select_candidate_records,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.sequence_groups import _sequence_groups

def export_ranked_frames(
    build_result: BuildResult,
    destination: Path | str,
    *,
    top_k: int,
    neighbor_radius: int = 1,
    metric_key: str = "export_priority_score",
    selection_mode: str = EXPORT_SELECTION_MODE_COUNT,
    top_percent: float = 10.0,
    percentile_threshold: float = 90.0,
) -> dict[str, object]:
    destination_path = Path(destination)
    destination_path.mkdir(parents=True, exist_ok=True)
    records = [record for record in build_result.records if record.summary is not None]
    if not records:
        raise ValueError("Nothing to export. Compute analytics first.")
    selected = list(select_candidate_records(
        replace(build_result, records=tuple(records)),
        metric_key=metric_key,
        selection_mode=selection_mode,
        top_k=top_k,
        top_percent=top_percent,
        percentile_threshold=percentile_threshold,
    ))
    percentile_map = compute_metric_percentiles(records, metric_key)
    sequence_groups = _sequence_groups(tuple(build_result.records))
    selected_keys = {record.key for record in selected}
    supplemental_keys: set[str] = set()
    radius = max(0, int(neighbor_radius))
    if radius > 0:
        for group in sequence_groups.values():
            key_to_index = {item.key: index for index, item in enumerate(group)}
            for record in selected:
                if record.key not in key_to_index:
                    continue
                center = key_to_index[record.key]
                for offset in range(-radius, radius + 1):
                    neighbor_index = center + offset
                    if neighbor_index < 0 or neighbor_index >= len(group):
                        continue
                    candidate = group[neighbor_index]
                    if candidate.key not in selected_keys:
                        supplemental_keys.add(candidate.key)

    export_records = {record.key: record for record in build_result.records if record.key in selected_keys | supplemental_keys}
    manifest: dict[str, object] = {
        "metric_key": metric_key,
        "selection_mode": selection_mode,
        "top_k": int(top_k),
        "top_percent": float(top_percent),
        "percentile_threshold": float(percentile_threshold),
        "selected_keys": [record.key for record in selected],
        "supplemental_keys": sorted(supplemental_keys, key=natural_sort_key),
        "scores": {record.key: metric_value_for_record(record, metric_key) for record in export_records.values()},
        "score_percentiles": {record.key: float(percentile_map.get(record.key, 0.0)) for record in export_records.values()},
        "reasons": {record.key: ([metric_key, "primary"] if record.key in selected_keys else [metric_key, "neighbor"]) for record in export_records.values()},
    }

    original_dir = destination_path / "original"
    gt_dir = destination_path / "gt"
    masks_root = destination_path / "models"
    probs_root = destination_path / "probabilities"
    for root in (original_dir, gt_dir, masks_root, probs_root):
        root.mkdir(parents=True, exist_ok=True)

    for key, record in export_records.items():
        normalized_name = key.replace("/", "__")
        if record.original_path:
            shutil.copy2(record.original_path, original_dir / Path(normalized_name).name)
        if record.gt_path:
            shutil.copy2(record.gt_path, gt_dir / Path(normalized_name).name)
        for model_id, path_text in record.model_mask_paths.items():
            model_dir = masks_root / model_id
            model_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path_text, model_dir / Path(normalized_name).name)
        for model_id, path_text in record.model_prob_paths.items():
            if not path_text:
                continue
            model_dir = probs_root / model_id
            model_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path_text, model_dir / Path(normalized_name).name)

    manifest_path = destination_path / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "selected_count": len(selected_keys),
        "supplemental_count": len(supplemental_keys),
        "manifest_path": str(manifest_path),
        "selected_keys": manifest["selected_keys"],
        "supplemental_keys": manifest["supplemental_keys"],
    }
