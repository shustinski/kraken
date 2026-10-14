"""augmentation_preview_dialog: close event."""
from __future__ import annotations
from PyQt6.QtWidgets import (
    QGroupBox,
    QVBoxLayout,
)
from neuralimage.augmentations import (
    PCBDefectAugmentor as PCBDefectAugmentor,
)
from neuralimage.configuration import (
    build_sem_segmentation_config,
)


class CloseEventMixin:
    def _build_shared_training_transform_groups(self) -> tuple[QGroupBox, QGroupBox]:
        return self._panel.sem_normalization_editor, self._panel.training_augmentation_editor

    def _attach_sidebar_widget(self, widget: QGroupBox, layout: QVBoxLayout) -> None:
        old_parent = widget.parentWidget()
        old_layout = old_parent.layout() if old_parent is not None else None
        index: int | None = None
        if old_layout is not None:
            index = old_layout.indexOf(widget)
            if index >= 0:
                old_layout.removeWidget(widget)
        layout.addWidget(widget)
        self._sidebar_restore.append((widget, old_parent, old_layout, index))

    def closeEvent(self, event) -> None:
        self._remote_preview_serial = getattr(self, "_remote_preview_serial", 0) + 1
        self._pending_preview = None
        for widget, old_parent, old_layout, index in reversed(self._sidebar_restore):
            current_parent = widget.parentWidget()
            if current_parent is not None:
                current_layout = current_parent.layout()
                if current_layout is not None:
                    current_layout.removeWidget(widget)
            if old_layout is not None and index is not None and index >= 0:
                widget.setParent(old_parent)
                if isinstance(old_layout, QVBoxLayout):
                    old_layout.insertWidget(index, widget)
                else:
                    old_layout.addWidget(widget)
            elif old_parent is not None:
                widget.setParent(old_parent)
        self._sidebar_restore.clear()
        super().closeEvent(event)

    def _current_sem_config(self):
        return build_sem_segmentation_config(self._panel.get_sem_segmentation_config())

    def _on_sem_pipeline_changed(self) -> None:
        self._prepared_arrays_cache.clear()
        self._cutter_length_cache.clear()
        self._sync_sem_pipeline_editor_visibility()
        self._rebuild_frame_plan()
        self._refresh_preview()

    def _sync_sem_pipeline_editor_visibility(self) -> None:
        self._panel._sync_sem_segmentation_controls()
