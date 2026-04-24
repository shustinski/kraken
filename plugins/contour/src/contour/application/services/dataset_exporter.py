"""Dataset export service.

Wraps :func:`contour.serializers.export_dataset_frame` so the widget can
offload dataset-export work to a small service object with no Qt dependency.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ...serializers import export_dataset_frame

if TYPE_CHECKING:
    from ...domain import PolygonData
    from ..processing import ImageProcessingState


@dataclass(frozen=True)
class DatasetExportResult:
    """Outcome of a single dataset-export call."""

    saved_files: dict[str, str]
    message_key: str | None = None
    message_kwargs: dict[str, object] | None = None

    @property
    def is_empty(self) -> bool:
        return not self.saved_files


def export_frame_to_dataset(
    *,
    dataset_directory: str,
    image_path: str,
    state: ImageProcessingState,
    polygons: list[PolygonData],
) -> DatasetExportResult:
    """Export a single frame and return the operation outcome.

    The caller is expected to translate ``message_key`` and emit the log line.
    """
    directory = dataset_directory.strip()
    if not directory:
        return DatasetExportResult(saved_files={}, message_key="dataset_directory_not_set_log")
    image_name = Path(image_path).name
    try:
        saved_files = export_dataset_frame(
            dataset_directory=directory,
            image_path=image_path,
            polygons=polygons,
            source_image=state.source_image,
        )
    except Exception as exc:  # pragma: no cover - surfaced to logs
        return DatasetExportResult(
            saved_files={},
            message_key="dataset_export_failed_log",
            message_kwargs={"image_name": image_name, "error": exc},
        )
    return DatasetExportResult(
        saved_files=saved_files,
        message_key="dataset_exported_log",
        message_kwargs={"image_name": image_name, "saved_files": saved_files},
    )


__all__ = ["DatasetExportResult", "export_frame_to_dataset"]


# The following symbol exists only so ``Callable`` type imports resolve.
_CallableT = Callable[..., object]
