"""Batch-processing controller service.

A thin façade over :class:`contour.batch_processor.BatchProcessor` that
owns the start/stop policy and the "is progress UI enabled" decision. It does
not depend on Qt widgets — the UI layer forwards user input and listens to the
underlying processor's signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...batch_processor import BatchProcessor
    from ..processing import ContourExtractionSettings, DisplaySettings, SaveOptions


@dataclass(frozen=True)
class BatchStartRequest:
    """Parameters accepted by :meth:`BatchController.start`."""

    image_paths: list[str]
    pipeline_config: dict[str, object]
    contour_settings: ContourExtractionSettings
    display_settings: DisplaySettings
    save_options: SaveOptions
    output_directory: str | None
    max_workers: int


class BatchController:
    """Wrap a :class:`BatchProcessor` with higher-level start/stop semantics."""

    def __init__(self, processor: BatchProcessor) -> None:
        self._processor = processor
        self._progress_enabled = False

    @property
    def processor(self) -> BatchProcessor:
        return self._processor

    @property
    def is_running(self) -> bool:
        return self._processor.is_running

    @property
    def progress_enabled(self) -> bool:
        """Whether per-image progress updates should be surfaced to the UI."""
        return self._progress_enabled

    def start(self, request: BatchStartRequest) -> bool:
        """Start batch processing. Returns ``False`` if the processor is already busy
        or if *request* has no images to process."""
        if self._processor.is_running:
            return False
        if not request.image_paths:
            return False
        self._progress_enabled = bool(request.output_directory and request.save_options.save_cif)
        self._processor.start(
            image_paths=list(request.image_paths),
            pipeline_config=request.pipeline_config,
            contour_settings=request.contour_settings,
            output_directory=request.output_directory,
            save_options=request.save_options,
            display_settings=request.display_settings,
            max_workers=request.max_workers,
        )
        return True

    def stop(self) -> None:
        self._processor.stop()


__all__ = ["BatchController", "BatchStartRequest"]
