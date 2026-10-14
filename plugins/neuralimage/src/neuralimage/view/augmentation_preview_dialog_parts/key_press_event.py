"""augmentation_preview_dialog: key press event."""
from __future__ import annotations
from PyQt6.QtCore import QEvent, QObject, Qt
from neuralimage.augmentations import (
    PCBDefectAugmentor as PCBDefectAugmentor,
)


class KeyPressEventMixin:
    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_visible_preview()

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key in (Qt.Key.Key_Up, Qt.Key.Key_Down):
            if self._navigate_frame(-1 if key == Qt.Key.Key_Up else 1):
                event.accept()
                return
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            event.accept()
            return
        super().keyPressEvent(event)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched is self.sample_list_widget and event.type() == QEvent.Type.KeyPress:
            key = event.key()
            if key in (Qt.Key.Key_Up, Qt.Key.Key_Down):
                if self._navigate_frame(-1 if key == Qt.Key.Key_Up else 1):
                    return True
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                return True
        return super().eventFilter(watched, event)
