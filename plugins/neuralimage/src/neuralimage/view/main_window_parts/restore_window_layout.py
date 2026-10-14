"""main_window: restore window layout."""
from __future__ import annotations
import os
from PyQt6.QtCore import QEvent, QSize, Qt, QTimer
from PyQt6.QtWidgets import (
    QDockWidget,
)

from neuralimage.view.main_window_parts.module_state import WINDOW_LAYOUT_STATE_VERSION

from neuralimage.view.main_window_parts.responsive_preview_label import _main_window_qsettings


class RestoreWindowLayoutMixin:
    def _managed_docks(self) -> tuple[QDockWidget, ...]:
        docks = [self.queue_dock, self.metrics_panel, self.log_dock]
        if self.settings_dock is not None:
            docks.append(self.settings_dock)
        return tuple(docks)

    def _save_window_layout(self) -> None:
        settings = _main_window_qsettings()
        settings.setValue('window_geometry', self.saveGeometry())
        window_size = self.normalGeometry().size() if self.isMaximized() or self.isFullScreen() else self.size()
        settings.setValue('window_size', window_size)
        settings.setValue('dock_layout', self.saveState(WINDOW_LAYOUT_STATE_VERSION))
        for dock in self._managed_docks():
            name = dock.objectName().strip()
            if not name:
                continue
            settings.beginGroup(f'docks/{name}')
            settings.setValue('size', dock.size())
            settings.setValue('area', int(self.dockWidgetArea(dock).value))
            settings.setValue('floating', dock.isFloating())
            settings.setValue('visible', not dock.isHidden())
            if dock.isFloating():
                settings.setValue('floating_geometry', dock.saveGeometry())
            else:
                settings.remove('floating_geometry')
            settings.endGroup()
        settings.sync()

    def _restore_window_layout(self) -> None:
        settings = _main_window_qsettings()
        geometry = settings.value('window_geometry')
        saved_window_size = settings.value('window_size', type=QSize)
        dock_layout = settings.value('dock_layout')
        self._restored_window_size: QSize | None = None
        if geometry is not None:
            if self.restoreGeometry(geometry):
                self._restored_window_size = self.size()
        if isinstance(saved_window_size, QSize) and saved_window_size.isValid():
            self._restored_window_size = saved_window_size

        state_restored = False
        if dock_layout is not None:
            state_restored = self.restoreState(dock_layout, WINDOW_LAYOUT_STATE_VERSION)
            if not state_restored:
                state_restored = self.restoreState(dock_layout)

        dock_sizes: dict[str, QSize] = {}
        for dock in self._managed_docks():
            name = dock.objectName().strip()
            if not name:
                continue
            settings.beginGroup(f'docks/{name}')
            saved_size = settings.value('size', type=QSize)
            saved_area = settings.value('area')
            floating = settings.value('floating', dock.isFloating(), type=bool)
            visible = settings.value('visible', not dock.isHidden(), type=bool)
            floating_geometry = settings.value('floating_geometry')
            settings.endGroup()

            if isinstance(saved_size, QSize) and saved_size.isValid():
                dock_sizes[name] = saved_size
            if not state_restored and saved_area is not None:
                try:
                    self.addDockWidget(Qt.DockWidgetArea(int(saved_area)), dock)
                except (TypeError, ValueError):
                    pass
            dock.setFloating(bool(floating))
            if floating and floating_geometry is not None:
                dock.restoreGeometry(floating_geometry)
            dock.setVisible(bool(visible))

        settings.sync()
        self._restored_dock_sizes = dock_sizes
        self._apply_restored_dock_sizes()

    def _apply_restored_dock_sizes(self) -> None:
        saved_sizes = getattr(self, '_restored_dock_sizes', {})
        if not saved_sizes:
            return
        for area in (
            Qt.DockWidgetArea.LeftDockWidgetArea,
            Qt.DockWidgetArea.RightDockWidgetArea,
        ):
            docks = [
                dock
                for dock in self._managed_docks()
                if not dock.isFloating() and self.dockWidgetArea(dock) == area and dock.objectName() in saved_sizes
            ]
            if docks:
                self.resizeDocks(docks, [saved_sizes[dock.objectName()].width() for dock in docks], Qt.Orientation.Horizontal)
        for area in (
            Qt.DockWidgetArea.TopDockWidgetArea,
            Qt.DockWidgetArea.BottomDockWidgetArea,
        ):
            docks = [
                dock
                for dock in self._managed_docks()
                if not dock.isFloating() and self.dockWidgetArea(dock) == area and dock.objectName() in saved_sizes
            ]
            if docks:
                self.resizeDocks(docks, [saved_sizes[dock.objectName()].height() for dock in docks], Qt.Orientation.Vertical)
        if self._restored_window_size is not None and not self.isMaximized() and not self.isFullScreen():
            self.resize(self._restored_window_size)

    def _finish_window_layout_restore(self) -> None:
        self._apply_restored_dock_sizes()
        self._window_layout_restore_pending = False

    def _schedule_window_layout_save(self, *_args) -> None:
        if self._window_layout_restore_pending or not self.isVisible():
            return
        if os.getenv('PYTEST_CURRENT_TEST') and not os.getenv('NEURALIMAGE_SETTINGS_DIR'):
            return
        self._window_layout_save_timer.start()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._window_layout_restore_pending:
            QTimer.singleShot(0, self._finish_window_layout_restore)

    def moveEvent(self, event) -> None:
        super().moveEvent(event)
        self._schedule_window_layout_save()

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange:
            self._schedule_window_layout_save()

    def eventFilter(self, watched, event) -> bool:
        if watched in self._managed_docks() and event.type() in {
            QEvent.Type.Move,
            QEvent.Type.Resize,
            QEvent.Type.Show,
            QEvent.Type.Hide,
        }:
            self._schedule_window_layout_save()
        return super().eventFilter(watched, event)
