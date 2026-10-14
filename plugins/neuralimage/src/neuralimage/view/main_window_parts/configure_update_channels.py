"""main_window: configure update channels."""
from __future__ import annotations
from PyQt6.QtGui import QAction, QActionGroup
from PyQt6.QtWidgets import (
    QLabel,
    QMenu,
    QMessageBox,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from neuralimage.lib.logging_policy import MAX_LOG_MESSAGES
from neuralimage.lib.ui_texts import get_ui_section

from neuralimage.view.main_window_parts.responsive_preview_label import _modal_dialogs_enabled


class ConfigureUpdateChannelsMixin:
    def _show_queue_context_menu(self, position) -> None:
        item = self.queue_list.itemAt(position)
        if item is None:
            return
        row = self.queue_list.row(item)
        if row < 0:
            return

        self.queue_list.setCurrentRow(row)
        texts = self._main_texts()
        status = ''
        if 0 <= row < len(self._queue_items):
            status = str(self._queue_items[row].get('status', ''))
        menu = QMenu(self.queue_list)
        report_action = None
        remove_action = None
        if status != 'in_progress':
            remove_action = menu.addAction(str(texts.get("queue_remove", "Удалить из очереди")))
        properties_action = menu.addAction(str(texts.get("queue_properties", "Свойства")))
        if status == 'finished_error':
            report_action = menu.addAction(str(texts.get("queue_report", "Report")))
        selected_action = menu.exec(self.queue_list.viewport().mapToGlobal(position))

        if remove_action is not None and selected_action is remove_action:
            self.queue_context_remove_requested.emit(row)
        elif selected_action is properties_action:
            self.queue_properties_requested.emit(row)
        elif report_action is not None and selected_action is report_action:
            self.queue_report_requested.emit(row)

    def _show_settings_page(self, page_key: str) -> None:
        self.show_settings_dock()
        if self.settings_dock is not None and hasattr(self.settings_dock, "show_settings_page"):
            self.settings_dock.show_settings_page(page_key)

    def _main_texts(self) -> dict[str, str]:
        texts = getattr(self, "_texts", None)
        return texts if isinstance(texts, dict) else get_ui_section("main_window")

    def _format_update_channel_label(self, channel: str) -> str:
        normalized_channel = str(channel or "").strip().lower()
        if not normalized_channel:
            normalized_channel = "stable"
        default_label = normalized_channel.replace("_", " ").title()
        return str(
            self._main_texts().get(
                f"update_channel_{normalized_channel}",
                default_label,
            )
        )

    def configure_update_channels(
        self,
        available_channels: list[str] | tuple[str, ...] | None,
        selected_channel: str | None,
    ) -> None:
        normalized_channels: list[str] = []
        for channel in available_channels or ("stable",):
            normalized = str(channel or "").strip().lower()
            if normalized and normalized not in normalized_channels:
                normalized_channels.append(normalized)
        if not normalized_channels:
            normalized_channels = ["stable"]

        resolved_selected = str(selected_channel or "").strip().lower()
        if resolved_selected not in normalized_channels:
            resolved_selected = normalized_channels[0]

        self._available_update_channels = normalized_channels
        self._selected_update_channel = resolved_selected
        self._update_channel_actions = {}

        if self._update_channel_menu is None:
            return

        self._update_channel_menu.clear()
        self._update_channel_action_group = QActionGroup(self)
        self._update_channel_action_group.setExclusive(True)

        for channel in normalized_channels:
            action = QAction(self._format_update_channel_label(channel), self)
            action.setCheckable(True)
            action.setData(channel)
            action.setChecked(channel == resolved_selected)
            action.triggered.connect(
                lambda checked=False, selected=channel: (
                    self.update_channel_selected.emit(selected) if checked else None
                )
            )
            self._update_channel_action_group.addAction(action)
            self._update_channel_menu.addAction(action)
            self._update_channel_actions[channel] = action

    def _show_info_message(self, text: str) -> None:
        if not _modal_dialogs_enabled():
            self.statusBar().showMessage(str(text))
            return
        QMessageBox.information(self, str(self._main_texts().get("info", "Информация")), text)

    def _show_warning_message(self, text: str) -> None:
        if not _modal_dialogs_enabled():
            self.statusBar().showMessage(str(text))
            return
        QMessageBox.warning(self, str(self._main_texts().get("warning", "Предупреждение")), text)

    def _format_validation_quality_text(self, iou: float, dice: float, f1: float) -> str:
        t = self._main_texts()
        template = str(
            t.get(
                "validation_quality_template",
                "IoU: {iou} | Dice: {dice} | F1: {f1}",
            )
        )
        return template.format(
            iou=f"{float(iou):.2%}",
            dice=f"{float(dice):.2%}",
            f1=f"{float(f1):.2%}",
        )

    def _format_performance_text(
        self,
        data_wait_ms: float,
        augmentation_ms: float,
        forward_ms: float,
        backward_ms: float,
        optimizer_ms: float,
        total_ms: float,
    ) -> str:
        t = self._main_texts()
        template = str(
            t.get(
                "performance_label_template",
                "Batch timing | data: {data_wait_ms:.1f} ms | aug: {augmentation_ms:.1f} ms | "
                "forward: {forward_ms:.1f} ms | "
                "backward: {backward_ms:.1f} ms | optimizer: {optimizer_ms:.1f} ms | total: {total_ms:.1f} ms",
            )
        )
        return template.format(
            data_wait_ms=float(data_wait_ms),
            augmentation_ms=float(augmentation_ms),
            forward_ms=float(forward_ms),
            backward_ms=float(backward_ms),
            optimizer_ms=float(optimizer_ms),
            total_ms=float(total_ms),
        )

    def _append_log(self, data):
        layout: QVBoxLayout = self.log_layout
        message_text = str(data)
        new_label = QLabel(message_text)
        layout.addWidget(new_label)
        self._last_status_message = message_text
        while layout.count() > MAX_LOG_MESSAGES:
            item = layout.takeAt(0)
            old_widget = item.widget() if item is not None else None
            if isinstance(old_widget, QWidget):
                old_widget.deleteLater()

        if isinstance(self.log_scroll, QScrollArea):
            vbar = self.log_scroll.verticalScrollBar()
            if vbar is not None:
                vbar.setValue(vbar.maximum())

    def append_with_delete_previous(self, data: str) -> None:
        layout: QVBoxLayout = self.log_layout
        count: int = layout.count()

        if count:
            item = layout.itemAt(count - 1)
            old_widget = item.widget() if item is not None else None
            if isinstance(old_widget, QWidget):
                old_widget.deleteLater()
                layout.removeWidget(old_widget)

        self._append_log(data)
