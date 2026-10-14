"""repository: compute build result metrics."""
from __future__ import annotations
from dataclasses import replace
from neuralimage.Validation_gradient_widget_lite.core.domain import (
    BuildResult,
    ComparisonMode,
    FrameRecord,
)

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.module_state import EPS

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.available_memory_bytes import BuildCancelledError

from neuralimage.Validation_gradient_widget_lite.core.repository_parts.load_frame_layers import (
    compute_comparison_score,
    load_frame_layers,
)

def compute_build_result_metrics(
    build_result: BuildResult,
    *,
    comparison_mode: ComparisonMode | None = None,
    display_metric: str = "relative",
    progress_callback=None,
    cancel_check=None,
) -> BuildResult:
    """Legacy-lite score-computation wrapper implemented on top of analytics."""

    mode = comparison_mode or build_result.options.comparison_mode
    normalized_records: list[FrameRecord] = []
    total = max(1, len(build_result.records))
    for index, record in enumerate(build_result.records, start=1):
        if cancel_check is not None and cancel_check():
            raise BuildCancelledError("Build cancelled")
        layers = load_frame_layers(record)
        score = compute_comparison_score(layers["first_binary"], layers["second_binary"], mode)
        normalized_records.append(
            replace(
                record,
                absolute_score=float(score),
                score=float(score),
                score_ready=True,
            )
        )
        if progress_callback is not None:
            progress_callback(index, total, record.key)
    if not normalized_records:
        return replace(build_result, options=replace(build_result.options, comparison_mode=mode), scores_computed=True)
    absolute_scores = [float(record.absolute_score or 0.0) for record in normalized_records]
    min_absolute = float(min(absolute_scores))
    max_absolute = float(max(absolute_scores))
    span = max(EPS, max_absolute - min_absolute)
    use_absolute = str(display_metric or "relative").lower() == "absolute"
    updated_records = []
    for record in normalized_records:
        absolute = float(record.absolute_score or 0.0)
        relative = 0.0 if abs(max_absolute - min_absolute) <= EPS else float((absolute - min_absolute) / span)
        display = absolute if use_absolute else relative
        updated_records.append(replace(record, score=float(display), relative_score=float(relative), score_ready=True))
    best_record = min(updated_records, key=lambda item: float(item.absolute_score or 0.0))
    return replace(
        build_result,
        records=tuple(updated_records),
        options=replace(build_result.options, comparison_mode=mode),
        min_score=min((record.score for record in updated_records), default=0.0),
        max_score=max((record.score for record in updated_records), default=0.0),
        scores_computed=True,
        best_match_key=best_record.key,
        min_absolute_score=min_absolute,
        max_absolute_score=max_absolute,
    )
