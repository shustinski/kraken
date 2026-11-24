"""Tree of loaded frame and vector sources with per-source removal.

Top level: frame sources.  Under each one: the vector sources whose CIFs
match its frames, with "matched / total".  Vector sources that match no
loaded frame are grouped under "Without frames".  Every removable row has an
✕ button; locked (Kraken) sources show a badge instead.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from PyQt6.QtCore import QPoint, QSize, Qt, QUrl, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QDesktopServices,
    QDragEnterEvent,
    QDropEvent,
    QGuiApplication,
    QIcon,
    QPainter,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from contour.application.sources import SOURCE_IMAGES, SOURCE_VECTORS, FrameSource, SourceRegistry

SOURCE_COLORS = (
    "#3B82F6",
    "#F59E0B",
    "#10B981",
    "#EC4899",
    "#8B5CF6",
    "#14B8A6",
    "#EF4444",
    "#84CC16",
)
_SOURCE_ID_ROLE = Qt.ItemDataRole.UserRole
_GROUP_ROLE = Qt.ItemDataRole.UserRole + 1

SOURCES_PANEL_STYLE = """
QLabel#sourcesTitle { font-weight: 600; }
QTreeWidget#sourcesTree { border: 1px solid palette(mid); border-radius: 6px; background: palette(base); }
QTreeWidget#sourcesTree::item { padding: 2px 0; }
QToolButton#sourceRemoveButton { border: none; padding: 0 4px; color: palette(placeholder-text); }
QToolButton#sourceRemoveButton:hover { color: #DC2626; }
QLabel#sourcesHint { color: palette(placeholder-text); }
QLabel#sourceBadge { border: 1px solid palette(mid); border-radius: 7px; padding: 0 6px; font-size: 10px; }
"""


def source_color(source: FrameSource) -> QColor:
    return QColor(SOURCE_COLORS[source.color_index % len(SOURCE_COLORS)])


def _chip_icon(color: QColor, *, diamond: bool) -> QIcon:
    pixmap = QPixmap(12, 12)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(color)
    if diamond:
        painter.translate(6, 6)
        painter.rotate(45)
        painter.drawRect(-3, -3, 7, 7)
    else:
        painter.drawRoundedRect(1, 1, 10, 10, 2, 2)
    painter.end()
    return QIcon(pixmap)


class SourcesPanel(QWidget):
    addRequested = pyqtSignal(str, str)  # kind, "folder" | "files"
    removeRequested = pyqtSignal(str)
    refreshRequested = pyqtSignal()
    sourceActivated = pyqtSignal(object)  # source id or None
    pathsDropped = pyqtSignal(list)

    def __init__(self, tr: Callable[..., str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._tr = tr
        self._registry: SourceRegistry | None = None
        self._scanning: set[str] = set()
        self._active_id: str | None = None
        self.setAcceptDrops(True)
        self.setStyleSheet(SOURCES_PANEL_STYLE)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        header = QHBoxLayout()
        header.setContentsMargins(0, 4, 0, 0)
        header.setSpacing(4)
        self.title_label = QLabel()
        self.title_label.setObjectName("sourcesTitle")
        header.addWidget(self.title_label)
        header.addStretch(1)
        self.add_button = QToolButton()
        self.add_button.setText("＋")
        self.add_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.add_menu = QMenu(self.add_button)
        self.add_button.setMenu(self.add_menu)
        self.add_button.setStyleSheet("QToolButton::menu-indicator { image: none; }")
        header.addWidget(self.add_button)
        self.refresh_button = QToolButton()
        self.refresh_button.setText("⟳")
        self.refresh_button.clicked.connect(self.refreshRequested.emit)
        header.addWidget(self.refresh_button)
        layout.addLayout(header)

        self.tree = QTreeWidget()
        self.tree.setObjectName("sourcesTree")
        self.tree.setColumnCount(3)
        self.tree.setHeaderHidden(True)
        self.tree.setRootIsDecorated(False)
        self.tree.setIndentation(14)
        self.tree.setUniformRowHeights(True)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tree.setIconSize(QSize(12, 12))
        self.tree.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_context_menu)
        self.tree.itemClicked.connect(self._on_item_clicked)
        header_view = self.tree.header()
        assert header_view is not None
        header_view.setStretchLastSection(False)
        header_view.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header_view.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header_view.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.tree)

        self.empty_widget = QWidget()
        empty_layout = QVBoxLayout(self.empty_widget)
        empty_layout.setContentsMargins(0, 4, 0, 4)
        self.empty_hint = QLabel()
        self.empty_hint.setObjectName("sourcesHint")
        self.empty_hint.setWordWrap(True)
        empty_layout.addWidget(self.empty_hint)
        buttons = QHBoxLayout()
        self.add_frames_button = QPushButton()
        self.add_frames_button.clicked.connect(lambda: self.addRequested.emit(SOURCE_IMAGES, "folder"))
        self.add_vectors_button = QPushButton()
        self.add_vectors_button.clicked.connect(lambda: self.addRequested.emit(SOURCE_VECTORS, "folder"))
        buttons.addWidget(self.add_frames_button)
        buttons.addWidget(self.add_vectors_button)
        empty_layout.addLayout(buttons)
        layout.addWidget(self.empty_widget)

        self.filter_hint = QLabel()
        self.filter_hint.setObjectName("sourcesHint")
        self.filter_hint.setWordWrap(True)
        self.filter_hint.linkActivated.connect(lambda _link: self._activate(None))
        self.filter_hint.hide()
        layout.addWidget(self.filter_hint)

        self.retranslate()
        self.rebuild(SourceRegistry())

    # ---------------------------------------------------------------- texts
    def retranslate(self) -> None:
        self.title_label.setText(self._tr("sources_title"))
        self.add_button.setToolTip(self._tr("sources_add_tooltip"))
        self.refresh_button.setToolTip(self._tr("refresh_files_button"))
        self.add_menu.clear()
        for kind, origin, key in (
            (SOURCE_IMAGES, "folder", "sources_add_image_folder"),
            (SOURCE_IMAGES, "files", "sources_add_image_files"),
            (SOURCE_VECTORS, "folder", "sources_add_vector_folder"),
            (SOURCE_VECTORS, "files", "sources_add_vector_files"),
        ):
            self.add_menu.addAction(self._tr(key), lambda k=kind, o=origin: self.addRequested.emit(k, o))
            if (kind, origin) == (SOURCE_IMAGES, "files"):
                self.add_menu.addSeparator()
        self.empty_hint.setText(self._tr("sources_empty_hint"))
        self.add_frames_button.setText(self._tr("sources_add_frames_button"))
        self.add_vectors_button.setText(self._tr("sources_add_vectors_button"))
        if self._registry is not None:
            self.rebuild(self._registry, scanning=self._scanning)

    # -------------------------------------------------------------- content
    def rebuild(self, registry: SourceRegistry, *, scanning: Iterable[str] = ()) -> None:
        self._registry = registry
        self._scanning = set(scanning)
        if self._active_id is not None and registry.get(self._active_id) is None:
            self._active_id = None
        self.tree.clear()
        report = registry.pairing()
        vector_sources = registry.sources(SOURCE_VECTORS)
        paired_vector_ids = {pair.vector_source_id for pair in report.pairs}
        for image in registry.sources(SOURCE_IMAGES):
            parent = self._add_source_item(None, image, self._image_count_text(image))
            for pair in report.pairs_for_image_source(image.id):
                vector = registry.get(pair.vector_source_id)
                if vector is None:
                    continue
                text = f"{pair.matched}/{pair.total}"
                if pair.matched < pair.total:
                    text = "⚠ " + text
                child = self._add_source_item(parent, vector, text)
                child.setToolTip(0, self._vector_tooltip(vector, report))
            parent.setExpanded(True)
        orphans = [vector for vector in vector_sources if vector.id not in paired_vector_ids]
        if orphans:
            group = QTreeWidgetItem(self.tree, ["◇ " + self._tr("sources_unpaired_group"), "", ""])
            group.setData(0, _GROUP_ROLE, True)
            group.setFlags(Qt.ItemFlag.ItemIsEnabled)
            group.setForeground(0, QBrush(self.palette().placeholderText().color()))
            for vector in orphans:
                self._add_source_item(group, vector, self._vector_count_text(vector))
            group.setExpanded(True)
        empty = registry.is_empty()
        self.tree.setVisible(not empty)
        self.empty_widget.setVisible(empty)
        self._fit_tree_height()
        self._refresh_selection()

    def _add_source_item(self, parent: QTreeWidgetItem | None, source: FrameSource, count: str) -> QTreeWidgetItem:
        label = self._source_label(source)
        if source.id in self._scanning:
            count = self._tr("path_status_scanning")
        item = QTreeWidgetItem(parent if parent is not None else self.tree, [label, count, ""])
        item.setIcon(0, _chip_icon(source_color(source), diamond=source.kind == SOURCE_VECTORS))
        item.setData(0, _SOURCE_ID_ROLE, source.id)
        item.setToolTip(0, self._source_tooltip(source))
        item.setTextAlignment(1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if count.startswith("⚠"):
            item.setForeground(1, QBrush(QColor("#D97706")))
        if source.locked:
            badge = QLabel("Kraken")
            badge.setObjectName("sourceBadge")
            badge.setToolTip(self._tr("path_locked_by_kraken"))
            self.tree.setItemWidget(item, 2, badge)
        else:
            remove = QToolButton()
            remove.setObjectName("sourceRemoveButton")
            remove.setText("✕")
            remove.setToolTip(
                self._tr("sources_remove_vectors_tooltip")
                if source.kind == SOURCE_VECTORS
                else self._tr("sources_remove_frames_tooltip")
            )
            remove.clicked.connect(lambda _checked=False, source_id=source.id: self.removeRequested.emit(source_id))
            self.tree.setItemWidget(item, 2, remove)
        return item

    def _source_label(self, source: FrameSource) -> str:
        if source.is_folder:
            return source.label
        count = len(source.effective_paths())
        key = "sources_files_label" if source.kind == SOURCE_IMAGES else "sources_vector_files_label"
        return self._tr(key, count=count)

    def _image_count_text(self, source: FrameSource) -> str:
        return str(len(source.effective_paths()))

    def _vector_count_text(self, source: FrameSource) -> str:
        return self._tr("sources_cif_count", count=len(source.effective_paths()))

    def _source_tooltip(self, source: FrameSource) -> str:
        lines = [source.root] if source.root else []
        if not source.is_folder:
            names = [Path(path).name for path in source.effective_paths()[:8]]
            lines.extend(names)
            if len(source.effective_paths()) > 8:
                lines.append("…")
        if source.excluded:
            lines.append(self._tr("sources_excluded_count", count=len(source.excluded)))
        return "\n".join(lines)

    def _vector_tooltip(self, source: FrameSource, report) -> str:
        lines = [self._source_tooltip(source)]
        overridden = int(report.overridden_counts.get(source.id, 0))
        if overridden:
            lines.append(self._tr("sources_overridden_count", count=overridden))
        unpaired = int(report.unpaired_vector_counts.get(source.id, 0))
        if unpaired:
            lines.append(self._tr("sources_unpaired_count", count=unpaired))
        if len(report.pairs_for_vector_source(source.id)) > 1:
            lines.append(self._tr("sources_shared_vectors_hint"))
        return "\n".join(line for line in lines if line)

    def _fit_tree_height(self) -> None:
        rows = 0
        iterator_items = [self.tree.topLevelItem(index) for index in range(self.tree.topLevelItemCount())]
        while iterator_items:
            item = iterator_items.pop()
            if item is None:
                continue
            rows += 1
            if item.isExpanded():
                iterator_items.extend(item.child(index) for index in range(item.childCount()))
        row_height = max(22, self.tree.sizeHintForRow(0)) if rows else 22
        visible_rows = min(max(rows, 1), 8)
        self.tree.setFixedHeight(visible_rows * row_height + 6)

    # ------------------------------------------------------------ selection
    def _on_item_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        source_id = item.data(0, _SOURCE_ID_ROLE)
        if not source_id:
            return
        self._activate(None if source_id == self._active_id else str(source_id))

    def _activate(self, source_id: str | None) -> None:
        self._active_id = source_id
        self._refresh_selection()
        self.sourceActivated.emit(source_id)

    def active_source_id(self) -> str | None:
        return self._active_id

    def _refresh_selection(self) -> None:
        self.tree.clearSelection()
        source = self._registry.get(self._active_id) if (self._registry and self._active_id) else None
        if source is None:
            self.filter_hint.hide()
            return
        for item in self._items():
            if item.data(0, _SOURCE_ID_ROLE) == source.id:
                item.setSelected(True)
                break
        self.filter_hint.setText(
            self._tr("sources_filter_hint", name=self._source_label(source))
            + f' <a href="all">{self._tr("sources_show_all")}</a>'
        )
        self.filter_hint.show()

    def _items(self) -> list[QTreeWidgetItem]:
        result: list[QTreeWidgetItem] = []
        stack = [self.tree.topLevelItem(index) for index in range(self.tree.topLevelItemCount())]
        while stack:
            item = stack.pop(0)
            if item is None:
                continue
            result.append(item)
            stack.extend(item.child(index) for index in range(item.childCount()))
        return result

    # --------------------------------------------------------- context menu
    def _show_context_menu(self, position: QPoint) -> None:
        item = self.tree.itemAt(position)
        source_id = item.data(0, _SOURCE_ID_ROLE) if item is not None else None
        source = self._registry.get(str(source_id)) if (self._registry and source_id) else None
        if source is None:
            return
        menu = QMenu(self)
        open_action = menu.addAction(self._tr("path_card_open_in_explorer"))
        copy_action = menu.addAction(self._tr("path_card_copy_path"))
        remove_action = None
        if not source.locked:
            menu.addSeparator()
            remove_action = menu.addAction(self._tr("sources_remove_action"))
        viewport = self.tree.viewport()
        assert viewport is not None
        chosen = menu.exec(viewport.mapToGlobal(position))
        if chosen is open_action and source.root and Path(source.root).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(source.root))
        elif chosen is copy_action:
            clipboard = QGuiApplication.clipboard()
            if clipboard is not None:
                clipboard.setText(source.root)
        elif remove_action is not None and chosen is remove_action:
            self.removeRequested.emit(source.id)

    # ------------------------------------------------------------ drag/drop
    def dragEnterEvent(self, event: QDragEnterEvent | None) -> None:
        if event is not None and event.mimeData() is not None and event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent | None) -> None:
        if event is None or event.mimeData() is None:
            return
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        if paths:
            event.acceptProposedAction()
            self.pathsDropped.emit(paths)


__all__ = ["SOURCE_COLORS", "SourcesPanel", "source_color"]
