from __future__ import annotations

import csv
import json
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class ExperimentRun:
    name: str
    seed: int
    wire_break_count: float
    false_bridge_count: float
    topology_violation_count: float
    boundary_f1: float
    boundary_iou: float
    hausdorff_distance: float
    dice: float
    iou: float
    runtime_seconds: float = 0.0
    peak_vram_mb: float = 0.0


@dataclass(frozen=True)
class ModelBenchmark:
    parameters: int
    trainable_parameters: int
    samples_per_second: float
    peak_vram_mb: float
    iterations: int
    effective_context_scale: float | None = None


def topology_first_key(run: ExperimentRun) -> tuple[float, ...]:
    """Lexicographic acceptance order; topology intentionally precedes Dice."""
    return (
        float(run.wire_break_count),
        float(run.false_bridge_count),
        float(run.topology_violation_count),
        -float(run.boundary_f1),
        -float(run.boundary_iou),
        float(run.hausdorff_distance),
        -float(run.dice),
        -float(run.iou),
    )


def rank_topology_first(runs: Sequence[ExperimentRun]) -> list[ExperimentRun]:
    return sorted(runs, key=topology_first_key)


def paired_bootstrap_delta(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 17,
) -> tuple[float, float, float]:
    """Return candidate-baseline mean delta and paired bootstrap interval."""
    baseline_values = np.asarray(baseline, dtype=np.float64)
    candidate_values = np.asarray(candidate, dtype=np.float64)
    if baseline_values.shape != candidate_values.shape or baseline_values.ndim != 1:
        raise ValueError('Paired bootstrap inputs must be one-dimensional and have equal shape.')
    if baseline_values.size == 0:
        raise ValueError('Paired bootstrap requires at least one pair.')
    deltas = candidate_values - baseline_values
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, deltas.size, size=(max(1, int(samples)), deltas.size))
    estimates = deltas[indices].mean(axis=1)
    tail = (1.0 - min(max(float(confidence), 0.0), 1.0)) / 2.0
    return (
        float(deltas.mean()),
        float(np.quantile(estimates, tail)),
        float(np.quantile(estimates, 1.0 - tail)),
    )


def _move_inputs(inputs: Any, device: torch.device) -> Any:
    if torch.is_tensor(inputs):
        return inputs.to(device)
    if isinstance(inputs, Mapping):
        return {key: _move_inputs(value, device) for key, value in inputs.items()}
    if isinstance(inputs, tuple):
        return tuple(_move_inputs(value, device) for value in inputs)
    return inputs


def benchmark_model(
    model: torch.nn.Module,
    inputs: Any,
    *,
    device: torch.device,
    warmup: int = 3,
    iterations: int = 20,
    effective_context_scale: float | None = None,
) -> ModelBenchmark:
    model = model.to(device).eval()
    prepared = _move_inputs(inputs, device)
    resolved_iterations = max(1, int(iterations))
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for _ in range(max(0, int(warmup))):
            model(prepared)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(resolved_iterations):
            model(prepared)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        elapsed = max(time.perf_counter() - started, 1e-9)
    local = prepared.get('local_image') if isinstance(prepared, Mapping) else prepared
    batch_size = int(local.shape[0]) if torch.is_tensor(local) and local.ndim > 0 else 1
    return ModelBenchmark(
        parameters=sum(parameter.numel() for parameter in model.parameters()),
        trainable_parameters=sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        samples_per_second=float(batch_size * resolved_iterations / elapsed),
        peak_vram_mb=(
            float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
            if device.type == 'cuda'
            else 0.0
        ),
        iterations=resolved_iterations,
        effective_context_scale=effective_context_scale,
    )


def write_experiment_report(
    output_dir: Path,
    runs: Sequence[ExperimentRun],
    *,
    dataset_manifest_hash: str,
    config_hash: str,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        'dataset_manifest_hash': str(dataset_manifest_hash),
        'config_hash': str(config_hash),
        'selection_rule': 'topology_first_v1',
        'runs': [asdict(run) for run in runs],
    }
    json_path = output_dir / 'experiment_report.json'
    csv_path = output_dir / 'experiment_runs.csv'
    json_handle = tempfile.NamedTemporaryFile(
        dir=output_dir, suffix='.tmp', mode='w', encoding='utf-8', delete=False
    )
    temporary_json = Path(json_handle.name)
    try:
        with json_handle:
            json.dump(payload, json_handle, indent=2, ensure_ascii=False)
            json_handle.flush()
            os.fsync(json_handle.fileno())
        os.replace(temporary_json, json_path)
    finally:
        temporary_json.unlink(missing_ok=True)
    handle = tempfile.NamedTemporaryFile(
        dir=output_dir, suffix='.tmp', mode='w', encoding='utf-8', newline='', delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            fieldnames = list(asdict(runs[0])) if runs else list(ExperimentRun.__dataclass_fields__)
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(asdict(run) for run in runs)
        os.replace(temporary, csv_path)
    finally:
        temporary.unlink(missing_ok=True)
    return json_path, csv_path
