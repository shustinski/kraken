"""Adapt Karakal analysis results to Kraken's shared frame-matrix contract."""
from __future__ import annotations

from dataclasses import dataclass

from kraken_core.frame_matrix.models import (
    MatrixBounds,
    MatrixItem,
    MatrixOrientation,
    MatrixSession,
    MatrixViewportRequest,
    MatrixViewportResult,
)
from kraken_core.frame_matrix.ports import CancellationToken

from ..core.analysis_modes import metric_visual_ratio
from ..core.domain import BuildResult, FrameRecord
from ..core.metric_keys import metric_higher_is_better
from ..ui.matrix_view import interpolate_gradient_color
from ..ui.ui_constants import DEFAULT_GRADIENT_NAME


@dataclass(frozen=True, slots=True)
class KarakalMatrixProjection:
    session: MatrixSession
    items: tuple[MatrixItem, ...]


def _record_position(record: FrameRecord, index: int, frames_per_row: int) -> tuple[int, int]:
    identity = record.identity
    if identity is not None and identity.tile_x is not None and identity.tile_y is not None:
        return int(identity.tile_x) + 1, int(identity.tile_y) + 1
    columns = max(1, int(frames_per_row))
    return index % columns + 1, index // columns + 1


def _record_goodness(record: FrameRecord, metric_key: str, score_view_mode: str, build_result: BuildResult) -> float | None:
    if not bool(getattr(record, "score_ready", False)):
        return None
    if score_view_mode != "absolute":
        return max(0.0, min(1.0, float(record.score)))
    if record.absolute_score is None:
        return None
    ratio = metric_visual_ratio(
        metric_key,
        float(record.absolute_score),
        point_match_radius=float(getattr(build_result.options, "point_match_radius", 3.0)),
        bce_score_cap=1.0,
    )
    if ratio is None:
        return None
    return max(0.0, min(1.0, float(ratio) if metric_higher_is_better(metric_key) else 1.0 - float(ratio)))


def project_build_result(
    build_result: BuildResult,
    *,
    metric_key: str,
    score_view_mode: str,
    gradient_name: str = DEFAULT_GRADIENT_NAME,
    frames_per_row: int = 100,
    namespace: str = "karakal-analysis",
) -> KarakalMatrixProjection:
    """Create a sparse, viewport-ready projection without copying project files."""

    items: list[MatrixItem] = []
    maximum_x = maximum_y = 1
    for index, record in enumerate(build_result.records):
        x, y = _record_position(record, index, frames_per_row)
        maximum_x = max(maximum_x, x)
        maximum_y = max(maximum_y, y)
        goodness = _record_goodness(record, metric_key, score_view_mode, build_result)
        excluded = bool(getattr(record, "excluded", False))
        status = "excluded" if excluded else "ready" if goodness is not None else "pending"
        metadata: dict[str, object] = {
            "metric_key": metric_key,
            "raw_metric_value": record.absolute_score,
            "goodness": goodness,
            "percentile": record.score_percentile,
            "excluded": excluded,
            "reference": str(record.key) == str(build_result.best_match_key or ""),
            "source_original": str(getattr(record, "original_path", "") or ""),
        }
        if goodness is not None:
            metadata["heatmap_value"] = goodness
            metadata["heatmap_color"] = interpolate_gradient_color(gradient_name, goodness).name()
        tooltip_parts = [str(record.display_name), f"metric: {metric_key}"]
        if record.absolute_score is not None:
            tooltip_parts.append(f"raw: {float(record.absolute_score):.6g}")
        if goodness is not None:
            tooltip_parts.append(f"quality: {goodness * 100.0:.2f}%")
        if record.score_percentile is not None:
            tooltip_parts.append(f"percentile: P{float(record.score_percentile):.1f}")
        items.append(
            MatrixItem(
                key=str(record.key),
                x=x,
                y=y,
                status=status,
                label=str(record.display_name),
                tooltip="\n".join(tooltip_parts),
                metadata=metadata,
            )
        )
    revision = f"{metric_key}:{score_view_mode}:{gradient_name}:{len(items)}"
    return KarakalMatrixProjection(
        session=MatrixSession(
            namespace=namespace,
            width=maximum_x,
            height=maximum_y,
            source_revision=revision,
            orientation=MatrixOrientation.Y_DOWN,
        ),
        items=tuple(items),
    )


class KarakalMatrixDataSource:
    """Serve one immutable Karakal projection through viewport requests."""

    def __init__(self, projection: KarakalMatrixProjection) -> None:
        self._projection = projection
        self._items_by_position = {(item.x, item.y): item for item in projection.items}

    @property
    def session(self) -> MatrixSession:
        return self._projection.session

    def load_viewport(
        self,
        request: MatrixViewportRequest,
        cancellation: CancellationToken | None = None,
    ) -> MatrixViewportResult:
        if cancellation is not None and cancellation.cancelled:
            return MatrixViewportResult(request=request, source_revision=self.session.source_revision)
        bounds = MatrixBounds(
            max(1, request.bounds.x1),
            max(1, request.bounds.y1),
            min(self.session.width, request.bounds.x2),
            min(self.session.height, request.bounds.y2),
        )
        items = tuple(
            item
            for coordinate in bounds.coordinates()
            if (item := self._items_by_position.get(coordinate)) is not None
        )
        return MatrixViewportResult(
            request=request,
            items=items,
            source_revision=self.session.source_revision,
        )


__all__ = [
    "KarakalMatrixDataSource",
    "KarakalMatrixProjection",
    "project_build_result",
]
