"""Kraken adapters for the shared frame-matrix data and asset ports."""

from __future__ import annotations

import threading
from pathlib import Path

from PyQt6.QtCore import QBuffer, QIODevice, Qt
from PyQt6.QtGui import QImage

from kraken_core.frame_matrix import (
    MatrixAggregate,
    MatrixAssetRef,
    MatrixBounds,
    MatrixItem,
    MatrixViewportResult,
)
from kraken_core.frame_matrix.ports import CancellationToken
from kraken_core.safe_files import open_regular_read


class KrakenMatrixDataSource:
    def __init__(
        self,
        service,
        *,
        project_id: str,
        layer_id: str = "",
        representation_ids: tuple[str, ...] = (),
        matrix_width: int = 1,
    ) -> None:
        self.service = service
        self.project_id = str(project_id)
        self._layer_id = str(layer_id)
        self._representation_ids = tuple(str(value) for value in representation_ids if str(value))
        self._matrix_width = max(1, int(matrix_width))
        self._lock = threading.RLock()

    def set_context(
        self,
        *,
        layer_id: str,
        representation_ids: tuple[str, ...],
        matrix_width: int | None = None,
    ) -> None:
        with self._lock:
            self._layer_id = str(layer_id)
            self._representation_ids = tuple(str(value) for value in representation_ids if str(value))
            if matrix_width is not None:
                self._matrix_width = max(1, int(matrix_width))

    def load_viewport(self, request, cancellation: CancellationToken | None = None) -> MatrixViewportResult:
        if cancellation is not None and cancellation.cancelled:
            return MatrixViewportResult(request)
        with self._lock:
            layer_id = self._layer_id
            representation_ids = self._representation_ids
            matrix_width = self._matrix_width
        if not layer_id:
            return MatrixViewportResult(request)
        bounds = request.bounds
        payload = self.service.matrix_viewport(
            self.project_id,
            layer_id=layer_id,
            representation_ids=representation_ids,
            x1=bounds.x1,
            y1=bounds.y1,
            x2=bounds.x2,
            y2=bounds.y2,
            lod=request.lod,
        )
        if cancellation is not None and cancellation.cancelled:
            return MatrixViewportResult(request)
        items: list[MatrixItem] = []
        for cell in payload.get("cells", ()):
            x, y = int(cell["x"]), int(cell["y"])
            number = (y - 1) * matrix_width + x
            artifact_sha256 = str(cell.get("sha256") or "")
            asset_source_key = str(
                cell.get("asset_sha256")
                or cell.get("asset_source_key")
                or ""
            )
            version = str(cell.get("artifact_version_id") or payload.get("revision") or "")
            asset_revision = str(cell.get("asset_revision") or version)
            tooltip = (
                "Отсутствует требуемый файл"
                if cell.get("missing")
                else (
                    f"Кадр ({x}, {y})\nСтатус: {cell.get('status', 'empty')}\n"
                    f"SHA-256: {artifact_sha256}\nВерсия: {version}"
                )
            )
            items.append(
                MatrixItem(
                    key=str(cell.get("frame_id") or f"{x}:{y}"),
                    x=x,
                    y=y,
                    status=str(cell.get("status") or "empty"),
                    label=f"{number:04d}",
                    tooltip=tooltip,
                    asset=(
                        MatrixAssetRef(
                            source_key=asset_source_key,
                            source_revision=asset_revision,
                            media_type=str(cell.get("asset_media_type") or "image/*"),
                            metadata={
                                "external_path": str(cell.get("asset_path") or ""),
                            },
                        )
                        if asset_source_key
                        else None
                    ),
                    metadata={
                        "artifact_version_id": version,
                        "asset_sha256": str(cell.get("asset_sha256") or ""),
                        "asset_revision": asset_revision,
                        "frame_id": str(cell.get("frame_id") or ""),
                        "missing": bool(cell.get("missing")),
                        "missing_representation_ids": tuple(
                            str(value)
                            for value in cell.get("missing_representation_ids", ())
                        ),
                        "modified_at": str(cell.get("modified_at") or ""),
                        "performer_color": str(cell.get("performer_color") or ""),
                        "performer_initials": str(cell.get("performer_initials") or ""),
                        "review_status": str(cell.get("review_status") or "not_checked"),
                        "quality": cell.get("quality"),
                    },
                )
            )
        aggregates = tuple(
            MatrixAggregate(
                bounds=MatrixBounds(**dict(value["bounds"])),
                materialized_count=int(value.get("materialized_count", 0)),
                status_counts={
                    str(status): int(count)
                    for status, count in dict(value.get("status_counts", {})).items()
                },
            )
            for value in payload.get("aggregates", ())
        )
        return MatrixViewportResult(
            request=request,
            items=tuple(items),
            aggregates=aggregates,
            source_revision=str(payload.get("revision") or ""),
        )


class KrakenMatrixAssetSource:
    def __init__(self, service, *, project_id: str) -> None:
        self.service = service
        self.project_id = str(project_id)

    def load_asset(
        self,
        reference: MatrixAssetRef,
        *,
        width: int,
        height: int,
        cancellation: CancellationToken | None = None,
    ) -> bytes:
        if cancellation is not None and cancellation.cancelled:
            return b""
        external_path = str(reference.metadata.get("external_path") or "")
        if external_path:
            path = Path(external_path)
            with open_regular_read(path, root=path.parent) as stream:
                source = stream.read()
        else:
            source = self.service.read_project_blob(self.project_id, reference.source_key)
        image = QImage.fromData(source)
        if image.isNull():
            raise ValueError(f"unsupported image asset: {reference.source_key}")
        scaled = image.scaled(
            max(1, int(width)),
            max(1, int(height)),
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
        if scaled.width() > width or scaled.height() > height:
            left = max(0, (scaled.width() - int(width)) // 2)
            top = max(0, (scaled.height() - int(height)) // 2)
            scaled = scaled.copy(left, top, max(1, int(width)), max(1, int(height)))
        output = QBuffer()
        output.open(QIODevice.OpenModeFlag.WriteOnly)
        if not scaled.save(output, "PNG"):
            raise ValueError("could not encode thumbnail")
        return output.data().data()


__all__ = ["KrakenMatrixAssetSource", "KrakenMatrixDataSource"]
