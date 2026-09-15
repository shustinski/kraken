"""Qt-free execution engine for manifest-driven mask analysis."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from kraken_core.analysis_protocol import AnalysisFrameResult, AnalysisMetricValue, AnalysisOutcome
from kraken_core.analysis_run_protocol import AnalysisExpression, AnalysisPartitionJobManifest


METRIC_REGISTRY_VERSION = "karakal.metrics.v1"
MASK_METRIC_KEYS = ("xor", "iou", "dice")


class AnalysisCancelled(RuntimeError):
    """Raised after a cooperative cancellation request."""


@dataclass(frozen=True, slots=True)
class AnalysisExecutionResult:
    outcome: AnalysisOutcome
    frames: tuple[AnalysisFrameResult, ...]
    message: str = ""


ProgressCallback = Callable[[int, int, str], None]
CancellationCheck = Callable[[], bool]


def _parameter_map(job: AnalysisPartitionJobManifest) -> dict[str, object]:
    return {parameter.key: parameter.value for parameter in job.parameters}


def _threshold(parameters: Mapping[str, object]) -> float:
    value = float(parameters.get("mask_threshold", parameters.get("threshold", 0.5)))
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("mask_threshold must be within 0..1")
    return value


def _load_mask(path: Path, *, threshold: float) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.ndim != 2 or image.size == 0:
        raise ValueError(f"Unable to decode grayscale image: {path.name}")
    return np.asarray(image >= round(threshold * 255.0), dtype=bool)


def _evaluate(expression: AnalysisExpression, masks: Mapping[str, np.ndarray]) -> np.ndarray:
    if expression.operation == "source":
        try:
            return masks[expression.source_key]
        except KeyError as exc:
            raise ValueError(f"Missing source binding: {expression.source_key}") from exc
    assert expression.left is not None and expression.right is not None
    left = _evaluate(expression.left, masks)
    right = _evaluate(expression.right, masks)
    if left.shape != right.shape:
        raise ValueError(f"Expression operands have different shapes: {left.shape} and {right.shape}")
    if expression.operation == "xor":
        return np.logical_xor(left, right)
    if expression.operation == "subtract":
        return np.logical_and(left, np.logical_not(right))
    if expression.operation == "compare":
        raise ValueError("compare can only be evaluated as the recipe root")
    raise ValueError(f"Unsupported expression operation: {expression.operation}")


def _comparison_operands(expression: AnalysisExpression, masks: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if expression.operation != "compare" or expression.left is None or expression.right is None:
        raise ValueError("Analysis recipe must contain a compare root")
    left = _evaluate(expression.left, masks)
    right = _evaluate(expression.right, masks)
    if left.shape != right.shape:
        raise ValueError(f"Comparison operands have different shapes: {left.shape} and {right.shape}")
    return left, right


def _metric_values(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    intersection = int(np.count_nonzero(first & second))
    first_area = int(np.count_nonzero(first))
    second_area = int(np.count_nonzero(second))
    union = first_area + second_area - intersection
    mismatch = int(np.count_nonzero(np.logical_xor(first, second)))
    pixel_count = max(1, int(first.size))
    return {
        "xor": float(mismatch / pixel_count),
        "iou": 1.0 if union == 0 else float(intersection / union),
        "dice": 1.0 if first_area + second_area == 0 else float(2 * intersection / (first_area + second_area)),
    }


def _metric_contract(key: str, raw_value: float) -> AnalysisMetricValue:
    higher_is_better = key != "xor"
    goodness = raw_value if higher_is_better else 1.0 - raw_value
    return AnalysisMetricValue(
        key=key,
        raw_value=raw_value,
        goodness=max(0.0, min(1.0, goodness)),
        unit="ratio",
        higher_is_better=higher_is_better,
    )


def _frame_masks(job: AnalysisPartitionJobManifest, frame_index: int, workspace: Path) -> dict[str, np.ndarray]:
    frame = job.frames[frame_index]
    threshold = _threshold(_parameter_map(job))
    masks: dict[str, np.ndarray] = {}
    for artifact in frame.artifacts:
        if artifact.binding_key not in job.recipe.expression.source_keys:
            continue
        path = workspace / Path(artifact.relative_path)
        if not path.is_file():
            raise ValueError(f"Input artifact does not exist: {artifact.relative_path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != artifact.sha256:
            raise ValueError(f"Input checksum mismatch: {artifact.relative_path}")
        masks[artifact.binding_key] = _load_mask(path, threshold=threshold)
    return masks


def run_analysis(
    job: AnalysisPartitionJobManifest,
    workspace: str | Path,
    output_dir: str | Path,
    progress: ProgressCallback | None = None,
    cancellation: CancellationCheck | None = None,
) -> AnalysisExecutionResult:
    """Calculate every configured mask metric without importing Qt state."""

    workspace_path = Path(workspace).resolve()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    metric_keys = job.recipe.metric_keys or MASK_METRIC_KEYS
    unsupported = sorted(set(metric_keys) - set(MASK_METRIC_KEYS))
    if unsupported:
        raise ValueError(f"Unsupported Karakal mask metrics: {', '.join(unsupported)}")

    results: list[AnalysisFrameResult] = []
    failed = 0
    total = len(job.frames)
    for index, frame in enumerate(job.frames):
        if cancellation is not None and cancellation():
            return AnalysisExecutionResult(AnalysisOutcome.CANCELLED, tuple(results), "Analysis cancelled")
        try:
            masks = _frame_masks(job, index, workspace_path)
            first, second = _comparison_operands(job.recipe.expression, masks)
            values = _metric_values(first, second)
            metrics = tuple(_metric_contract(key, values[key]) for key in metric_keys)
            result = AnalysisFrameResult(frame.frame_id, frame.x, frame.y, "ready", metrics)
        except (OSError, ValueError) as exc:
            failed += 1
            result = AnalysisFrameResult(
                frame.frame_id,
                frame.x,
                frame.y,
                "not_computed",
                (),
                message=str(exc),
            )
        results.append(result)
        if progress is not None:
            progress(index + 1, total, frame.frame_id)

    if failed == total:
        outcome = AnalysisOutcome.FAILED
    elif failed:
        outcome = AnalysisOutcome.PARTIAL
    else:
        outcome = AnalysisOutcome.SUCCEEDED
    message = "" if not failed else f"{failed} of {total} frames were not computed"
    return AnalysisExecutionResult(outcome, tuple(results), message)


def render_analysis_map(
    job: AnalysisPartitionJobManifest,
    frame_id: str,
    workspace: str | Path,
    cache_dir: str | Path,
) -> Path:
    """Render and cache an RGB difference map for one selected frame."""

    try:
        frame_index = next(index for index, frame in enumerate(job.frames) if frame.frame_id == frame_id)
    except StopIteration as exc:
        raise ValueError(f"Frame is not part of the analysis partition: {frame_id}") from exc
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.sha256(f"{job.fingerprint}:{frame_id}:comparison-map.v1".encode()).hexdigest()
    target = cache_root / f"{cache_key}.png"
    if target.is_file():
        return target

    masks = _frame_masks(job, frame_index, Path(workspace).resolve())
    first, second = _comparison_operands(job.recipe.expression, masks)
    canvas = np.zeros((*first.shape, 3), dtype=np.uint8)
    canvas[first & second] = (255, 255, 255)
    canvas[first & ~second] = (0, 0, 255)
    canvas[~first & second] = (255, 0, 0)
    temporary = target.with_suffix(".tmp.png")
    if not cv2.imwrite(str(temporary), canvas):
        raise OSError(f"Unable to render analysis map: {target}")
    temporary.replace(target)
    return target


__all__ = [
    "AnalysisCancelled",
    "AnalysisExecutionResult",
    "MASK_METRIC_KEYS",
    "METRIC_REGISTRY_VERSION",
    "render_analysis_map",
    "run_analysis",
]
