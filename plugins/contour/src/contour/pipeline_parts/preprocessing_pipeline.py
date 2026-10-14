"""pipeline: preprocessing pipeline."""
from __future__ import annotations
from time import perf_counter
from typing import Any
import numpy as np
from ..application.preview_cancellation import raise_if_preview_cancelled
from ..application.processing import PipelineStepConfig
from contour.i18n import operation_name, tr
from contour.utils import ensure_uint8

from contour.pipeline_parts.operation_descriptor import get_operation_descriptor

from contour.pipeline_parts.watershed_split import _operation_timing_bucket

class PreprocessingPipeline:
    def __init__(self, steps: list[PipelineStepConfig] | None = None) -> None:
        self.steps = [step.clone() for step in (steps or [])]

    def apply(
        self,
        image: np.ndarray,
        *,
        timing_callback: OperationTimingCallback | None = None,
    ) -> np.ndarray:
        result = ensure_uint8(image)
        for index, step in enumerate(self.steps):
            if not step.enabled:
                continue
            descriptor = get_operation_descriptor(step.operation)
            started_at = perf_counter() if timing_callback is not None else 0.0
            try:
                result = descriptor.handler(result, dict(step.parameters))
                result = ensure_uint8(result)
            except Exception as exc:
                raise RuntimeError(
                    tr(
                        "pipeline_step_failed",
                        index=index + 1,
                        step=operation_name(descriptor.type_name, default=step.name, language=None),
                        error=exc,
                    )
                ) from exc
            if timing_callback is not None:
                timing_callback(
                    index,
                    descriptor.type_name,
                    (perf_counter() - started_at) * 1000.0,
                )
            raise_if_preview_cancelled()
        return result

    def apply_with_timing(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        """Apply the same configured steps while aggregating coarse timing buckets.

        This is intentionally a thin measurement wrapper over the existing
        handlers so UI-configured processing semantics remain unchanged.
        """
        result = ensure_uint8(image)
        timings = {"threshold_ms": 0.0, "morphology_ms": 0.0, "postprocessing_ms": 0.0}
        for index, step in enumerate(self.steps):
            if not step.enabled:
                continue
            descriptor = get_operation_descriptor(step.operation)
            bucket = _operation_timing_bucket(step.operation)
            started = perf_counter()
            try:
                result = descriptor.handler(result, dict(step.parameters))
                result = ensure_uint8(result)
            except Exception as exc:
                raise RuntimeError(
                    tr(
                        "pipeline_step_failed",
                        index=index + 1,
                        step=operation_name(descriptor.type_name, default=step.name, language=None),
                        error=exc,
                    )
                ) from exc
            timings[bucket] = timings.get(bucket, 0.0) + (perf_counter() - started) * 1000.0
            raise_if_preview_cancelled()
        return result, timings

    def to_dict(self) -> dict[str, Any]:
        return {"steps": [step.to_dict() for step in self.steps]}

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> PreprocessingPipeline:
        payload = payload or {}
        steps = [PipelineStepConfig.from_dict(item) for item in payload.get("steps", [])]
        return cls(steps)

    @staticmethod
    def create_step(operation_name: str) -> PipelineStepConfig:
        descriptor = get_operation_descriptor(operation_name)
        return PipelineStepConfig(
            operation=descriptor.type_name,
            name=descriptor.display_name,
            enabled=True,
            parameters=descriptor.default_parameters(),
        )
