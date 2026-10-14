"""general_neural_handler: deterministic validation split."""
from __future__ import annotations
from pathlib import Path
from kraken_core.asset_source import AssetSource

from neuralimage.model.general_neural_handler_parts.module_state import _VALIDATION_SPLIT_SEED

from neuralimage.model.general_neural_handler_parts.estimate_label_foreground_ratio import (
    _estimate_label_foreground_ratio,
    _label_ratio_bucket,
    _stable_sample_sort_key,
)

def _deterministic_validation_split(
    samples: list[tuple[Path, Path]],
    *,
    val_count: int,
    seed: int = _VALIDATION_SPLIT_SEED,
    asset_source: AssetSource | None = None,
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    ordered_samples = sorted(samples, key=lambda sample: _stable_sample_sort_key(sample, seed=seed))
    total_count = len(ordered_samples)
    resolved_val_count = max(0, min(int(val_count), max(0, total_count - 1)))
    if resolved_val_count <= 0:
        return ordered_samples, []

    try:
        bucket_to_samples: dict[int, list[tuple[Path, Path]]] = {}
        for sample in ordered_samples:
            ratio = (
                _estimate_label_foreground_ratio(sample[1], asset_source)
                if asset_source is not None
                else _estimate_label_foreground_ratio(sample[1])
            )
            bucket = _label_ratio_bucket(ratio)
            bucket_to_samples.setdefault(bucket, []).append(sample)
    except Exception:
        if asset_source is not None:
            raise
        return ordered_samples[resolved_val_count:], ordered_samples[:resolved_val_count]

    bucket_counts = {bucket: len(bucket_samples) for bucket, bucket_samples in bucket_to_samples.items()}
    assigned_val_counts = {bucket: 0 for bucket in bucket_to_samples}
    remainders: list[tuple[float, int]] = []
    for bucket, bucket_size in bucket_counts.items():
        ideal = (bucket_size * resolved_val_count) / max(1, total_count)
        assigned = min(bucket_size, int(ideal))
        assigned_val_counts[bucket] = assigned
        remainders.append((ideal - assigned, bucket))

    remaining_slots = resolved_val_count - sum(assigned_val_counts.values())
    for _remainder, bucket in sorted(remainders, key=lambda item: (-item[0], item[1])):
        if remaining_slots <= 0:
            break
        if assigned_val_counts[bucket] >= bucket_counts[bucket]:
            continue
        assigned_val_counts[bucket] += 1
        remaining_slots -= 1

    validation_samples: list[tuple[Path, Path]] = []
    training_candidates: list[tuple[Path, Path]] = []
    for bucket, bucket_samples in sorted(bucket_to_samples.items()):
        take = min(assigned_val_counts[bucket], len(bucket_samples))
        validation_samples.extend(bucket_samples[:take])
        training_candidates.extend(bucket_samples[take:])

    if len(validation_samples) < resolved_val_count:
        missing = resolved_val_count - len(validation_samples)
        validation_samples.extend(training_candidates[:missing])
        training_candidates = training_candidates[missing:]

    validation_keys = {_stable_sample_sort_key(sample, seed=seed) for sample in validation_samples}
    training_samples = [sample for sample in ordered_samples if _stable_sample_sort_key(sample, seed=seed) not in validation_keys]
    validation_samples = [sample for sample in ordered_samples if _stable_sample_sort_key(sample, seed=seed) in validation_keys]
    return training_samples, validation_samples
