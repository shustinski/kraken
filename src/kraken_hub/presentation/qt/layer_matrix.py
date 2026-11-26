"""Frame matrix of a Kraken project layer and the actions it offers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from kraken_core.frame_matrix.actions import FrameMatrixAction
from kraken_core.frame_matrix.qt import FrameContext, FrameMatrixWidget

_SELECTION_LIMIT = 10_000


class LayerMatrixActionHost(Protocol):
    """Application state the layer actions depend on."""

    def can_assign_work(self) -> bool: ...

    def vector_file_for(self, x: int, y: int) -> object | None: ...


def _selected_count(context: FrameContext) -> int:
    return len(tuple(context.selection.coordinates(maximum=_SELECTION_LIMIT)))


def _export_title(context: FrameContext) -> str:
    count = _selected_count(context)
    return "Выгрузить файлы кадра…" if count == 1 else f"Выгрузить файлы выбранных кадров ({count})…"


class LayerFrameMatrixWidget(FrameMatrixWidget):
    """Kraken layer matrix: export, work assignment and frame/vector properties."""

    def __init__(self, *args, **kwargs) -> None:
        self._action_host: LayerMatrixActionHost | None = None
        super().__init__(*args, **kwargs)

    def set_action_host(self, host: LayerMatrixActionHost | None) -> None:
        self._action_host = host

    def frame_actions(self, context: FrameContext) -> Sequence[FrameMatrixAction]:
        return (
            FrameMatrixAction("export_files", _export_title),
            FrameMatrixAction("assign_work", "Выдать работу…", self._can_assign_work),
            FrameMatrixAction("vector_properties", "Свойства вектора…", self._has_vector_file),
            FrameMatrixAction("frame_properties", "Свойства кадра…", lambda ctx: _selected_count(ctx) == 1),
        )

    def _can_assign_work(self, context: FrameContext) -> bool:
        host = self._action_host
        return bool(_selected_count(context) and host is not None and host.can_assign_work())

    def _has_vector_file(self, context: FrameContext) -> bool:
        host = self._action_host
        if host is None or _selected_count(context) != 1:
            return False
        return host.vector_file_for(context.x, context.y) is not None


__all__ = ["LayerFrameMatrixWidget", "LayerMatrixActionHost"]
