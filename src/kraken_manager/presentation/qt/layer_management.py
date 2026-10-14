"""Modeless layer manager and provenance graph for a project workspace."""

from __future__ import annotations
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable
from PyQt6.QtCore import QLineF, QPointF, QRectF, QSettings, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSceneMouseEvent,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QHeaderView,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from .models import LayerListItem

from .layer_management_parts.module_state import (
    _EVENT_TYPE_LABELS,
    _FIELD_LABELS,
    _STATE_LABELS,
    _ROLE_LABELS,
    _ID_FIELD_SUFFIXES,
)
from .layer_management_parts.construction import ConstructionMixin
from .layer_management_parts.action_tab import ActionTabMixin
from .layer_management_parts.format_scalar import (
    PipelineNode,
    PipelineLane,
    LayerPipelineSnapshot,
    ObjectHistoryEntry,
    ObjectPropertiesSnapshot,
    _display_value,
    _local_timestamp,
    _event_type_label,
    _is_uuid_like,
    _format_bytes,
    _format_scalar,
    _field_label,
    _should_skip_key,
    _append_lines,
    _append_mapping_fields,
)
from .layer_management_parts.selection_summary import _selection_summary
from .layer_management_parts.format_history_payload import _format_history_payload
from .layer_management_parts.layer_order_list import LayerOrderList
from .layer_management_parts.pipeline_scene import PipelineScene
from .layer_management_parts.pipeline_node_item import PipelineNodeItem
from .layer_management_parts.pipeline_graph_view import PipelineGraphView
from .layer_management_parts.layer_manager_dialog import LayerManagerDialog

class ObjectPropertiesDialog(ConstructionMixin, ActionTabMixin, QDialog):
    """Read-only object metadata and event history."""

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    _EVENT_TYPE_LABELS,
    _FIELD_LABELS,
    _STATE_LABELS,
    _ROLE_LABELS,
    _ID_FIELD_SUFFIXES,
    PipelineNode,
    PipelineLane,
    LayerPipelineSnapshot,
    ObjectHistoryEntry,
    ObjectPropertiesSnapshot,
    _display_value,
    _local_timestamp,
    _event_type_label,
    _is_uuid_like,
    _format_bytes,
    _format_scalar,
    _field_label,
    _should_skip_key,
    _append_lines,
    _append_mapping_fields,
    _selection_summary,
    _format_history_payload,
    LayerOrderList,
    PipelineScene,
    PipelineNodeItem,
    PipelineGraphView,
    LayerManagerDialog,
)

__all__ = [
    "LayerManagerDialog",
    "LayerPipelineSnapshot",
    "ObjectHistoryEntry",
    "ObjectPropertiesDialog",
    "ObjectPropertiesSnapshot",
    "PipelineLane",
    "PipelineNode",
]
