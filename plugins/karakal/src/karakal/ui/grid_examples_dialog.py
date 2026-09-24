"""Window that lists marked cell examples, one tab per defect type."""

from __future__ import annotations

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .ui_constants import GRID_INSPECTION_ERROR_TYPE_OPTIONS


class GridExamplesDialog(QDialog):
    examplesChanged = pyqtSignal(list)

    def __init__(self, translator, examples: list[dict[str, object]], image: QPixmap | None, parent=None) -> None:
        super().__init__(parent)
        self._t = translator
        self._examples = [dict(item) for item in examples]
        self._image = image if image is not None and not image.isNull() else QPixmap()
        self.setWindowTitle(translator("grid_examples.window"))
        self.setModal(False)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self._tabs = QTabWidget(self)
        root = QVBoxLayout(self)
        root.addWidget(self._tabs)
        close_button = QPushButton(translator("grid_tuning.close"), self)
        close_button.clicked.connect(self.close)
        root.addWidget(close_button)
        self.resize(520, 420)
        self._rebuild()

    def set_examples(self, examples: list[dict[str, object]], image: QPixmap | None = None) -> None:
        self._examples = [dict(item) for item in examples]
        if image is not None and not image.isNull():
            self._image = image
        current = self._tabs.currentIndex()
        self._rebuild()
        if 0 <= current < self._tabs.count():
            self._tabs.setCurrentIndex(current)

    def _labels(self) -> list[tuple[str, str]]:
        return [("good", self._t("grid_examples.good")), ("ignore", self._t("grid_examples.ignore"))] + [
            (str(error_type), self._t(label_key)) for label_key, error_type in GRID_INSPECTION_ERROR_TYPE_OPTIONS
        ]

    def _rebuild(self) -> None:
        self._tabs.clear()
        for label, title in self._labels():
            rows = [item for item in self._examples if str(item.get("label") or "") == label]
            page = QWidget(self._tabs)
            layout = QVBoxLayout(page)
            listing = QListWidget(page)
            if not rows:
                empty = QListWidgetItem(self._t("grid_examples.empty"))
                empty.setFlags(Qt.ItemFlag.NoItemFlags)
                listing.addItem(empty)
            for index, example in enumerate(rows, start=1):
                item = QListWidgetItem()
                row = self._example_row(example, index)
                item.setSizeHint(QSize(0, 88))
                listing.addItem(item)
                listing.setItemWidget(item, row)
            layout.addWidget(listing)
            self._tabs.addTab(page, f"{title} ({len(rows)})")

    def _example_row(self, example: dict[str, object], index: int) -> QWidget:
        bbox = tuple(example.get("bbox", ()))
        width = int(bbox[2]) if len(bbox) >= 4 else 0
        height = int(bbox[3]) if len(bbox) >= 4 else 0
        row = QWidget(self)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(4, 4, 4, 4)
        thumb = QLabel(row)
        thumb.setFixedSize(72, 72)
        thumb.setPixmap(self._crop(bbox).scaled(72, 72, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        layout.addWidget(thumb)
        text = self._t("grid_examples.item", index=index, width=width, height=height)
        if not example.get("features"):
            text = f"{text}\n{self._t('grid_examples.no_features')}"
        layout.addWidget(QLabel(text, row), stretch=1)
        delete_button = QPushButton(self._t("grid_examples.delete"), row)
        delete_button.clicked.connect(lambda *_args, target=example: self._delete(target))
        layout.addWidget(delete_button)
        return row

    def _crop(self, bbox: tuple) -> QPixmap:
        if self._image.isNull() or len(bbox) < 4:
            pixmap = QPixmap(72, 72)
            pixmap.fill(Qt.GlobalColor.darkGray)
            return pixmap
        x, y, width, height = (int(value) for value in bbox[:4])
        return self._image.copy(max(0, x), max(0, y), max(1, width), max(1, height))

    def _delete(self, example: dict[str, object]) -> None:
        label = str(example.get("label") or "")
        bbox = tuple(example.get("bbox", ()))
        self._examples = [
            item
            for item in self._examples
            if not (str(item.get("label") or "") == label and tuple(item.get("bbox", ())) == bbox)
        ]
        self.examplesChanged.emit(list(self._examples))
        self._rebuild()
