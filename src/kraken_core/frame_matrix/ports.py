"""Ports used by the frame matrix without depending on Qt or infrastructure."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol

from .models import MatrixAssetRef, MatrixItem, MatrixViewportRequest, MatrixViewportResult


class CancellationToken(Protocol):
    @property
    def cancelled(self) -> bool: ...


class MatrixDataSource(Protocol):
    def load_viewport(
        self,
        request: MatrixViewportRequest,
        cancellation: CancellationToken | None = None,
    ) -> MatrixViewportResult: ...


class MatrixAssetSource(Protocol):
    def load_asset(
        self,
        reference: MatrixAssetRef,
        *,
        width: int,
        height: int,
        cancellation: CancellationToken | None = None,
    ) -> bytes: ...


class MatrixLayout(Protocol):
    def position_for(self, item: MatrixItem) -> tuple[int, int]: ...

    def item_key_at(self, row: int, column: int) -> str | None: ...


class MatrixLayerRenderer(Protocol):
    """Qt adapters may pass painter-specific context through ``context``."""

    @property
    def fingerprint(self) -> str: ...

    def render(self, item: MatrixItem, context: Any) -> None: ...


class MatrixLodPolicy(Protocol):
    def lod_for_zoom(self, zoom: float) -> int: ...

    def visible_layers(self, zoom: float) -> Iterable[str]: ...


__all__ = [
    "CancellationToken",
    "MatrixAssetSource",
    "MatrixDataSource",
    "MatrixLayerRenderer",
    "MatrixLayout",
    "MatrixLodPolicy",
]
