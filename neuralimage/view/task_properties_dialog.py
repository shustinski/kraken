from __future__ import annotations

from dataclasses import fields, replace
from typing import Any

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from application.dto import MainWindowState, SettingsState, clone_main_window_state
from lib.data_interfaces import normalize_multi_gpu_mode, normalize_patch_batch_sync_mode, normalize_work_mode
from lib.ui_texts import get_ui_section


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _fallback_field_label(name: str) -> str:
    words = [part for part in str(name).split('_') if part]
    if not words:
        return str(name)
    return ' '.join(words).capitalize()


class TaskPropertiesDialog(QDialog):
    restore_requested: pyqtSignal = pyqtSignal(object, object)

    def __init__(
        self,
        *,
        task_id: int,
        status: str,
        paused: bool,
        main_window_state: MainWindowState,
        settings_state: SettingsState,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._task_id = int(task_id)
        self._status = str(status)
        self._paused = bool(paused)
        self._main_window_state = clone_main_window_state(main_window_state)
        self._settings_state = replace(settings_state)
        self._texts = get_ui_section('task_properties_dialog')
        webui_texts = get_ui_section('webui')
        self._main_form_texts = _as_dict(webui_texts.get('main_form'))
        self._settings_form_texts = _as_dict(webui_texts.get('settings_form'))
        self._main_labels = _as_dict(self._main_form_texts.get('labels'))
        self._main_tooltips = _as_dict(self._main_form_texts.get('tooltips'))
        self._settings_labels = _as_dict(self._settings_form_texts.get('labels'))
        self._settings_tooltips = _as_dict(self._settings_form_texts.get('tooltips'))
        self._main_choice_maps = {
            'work_mode': _as_dict(self._main_form_texts.get('work_modes')),
        }
        self._settings_choice_maps = {
            key: _as_dict(value)
            for key, value in _as_dict(self._settings_form_texts.get('choices')).items()
            if isinstance(value, dict)
        }

        self.setWindowTitle(str(self._texts.get('window_title', 'Свойства задачи')))
        self.resize(860, 680)

        self._setup_ui()
        self._populate_tree()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        task_title = str(
            self._texts.get('task_title_template', 'Задача #{task_id}')
        ).format(task_id=self._task_id)
        self._title_label = QLabel(task_title, self)
        self._title_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._title_label.setStyleSheet('font-size: 16px; font-weight: 600;')
        layout.addWidget(self._title_label)

        self.tree = QTreeWidget(self)
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(
            [
                str(self._texts.get('parameter_column', 'Параметр')),
                str(self._texts.get('value_column', 'Значение')),
            ]
        )
        self.tree.setAlternatingRowColors(True)
        self.tree.setUniformRowHeights(True)
        header = self.tree.header()
        if header is not None:
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.tree)

        self._buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        self.restore_button = QPushButton(
            str(self._texts.get('restore_button', 'Восстановить')),
            self,
        )
        self._buttons.addButton(self.restore_button, QDialogButtonBox.ButtonRole.ActionRole)
        self._buttons.rejected.connect(self.reject)
        self.restore_button.clicked.connect(self._on_restore_clicked)
        layout.addWidget(self._buttons)

    def _populate_tree(self) -> None:
        self.tree.clear()
        self._populate_queue_section()
        self._populate_state_section(
            title=str(self._texts.get('main_section', 'Главное окно')),
            state=self._main_window_state,
            labels=self._main_labels,
            tooltips=self._main_tooltips,
            choice_maps=self._main_choice_maps,
            state_name='main',
        )
        self._populate_state_section(
            title=str(self._texts.get('settings_section', 'Настройки')),
            state=self._settings_state,
            labels=self._settings_labels,
            tooltips=self._settings_tooltips,
            choice_maps=self._settings_choice_maps,
            state_name='settings',
        )
        self.tree.expandAll()

    def _populate_queue_section(self) -> None:
        labels = _as_dict(self._texts.get('queue_labels'))
        section = self._create_section(str(self._texts.get('queue_section', 'Очередь')))
        self._add_value_item(
            section,
            str(labels.get('task_id', 'ID задачи')),
            str(self._task_id),
        )
        self._add_value_item(
            section,
            str(labels.get('status', 'Статус')),
            self._format_status(self._status),
        )
        self._add_value_item(
            section,
            str(labels.get('paused', 'На паузе')),
            self._format_bool(self._paused),
        )

    def _populate_state_section(
        self,
        *,
        title: str,
        state: Any,
        labels: dict[str, Any],
        tooltips: dict[str, Any],
        choice_maps: dict[str, dict[str, Any]],
        state_name: str,
    ) -> None:
        section = self._create_section(title)
        for field in fields(state):
            if state_name == 'main' and field.name == 'mode_state':
                continue
            raw_value = getattr(state, field.name)
            label = str(labels.get(field.name, _fallback_field_label(field.name)))
            tooltip = str(tooltips.get(field.name, ''))
            value = self._format_field_value(
                state_name=state_name,
                field_name=field.name,
                value=raw_value,
                choice_maps=choice_maps,
            )
            self._add_value_item(section, label, value, tooltip=tooltip)

    def _create_section(self, title: str) -> QTreeWidgetItem:
        item = QTreeWidgetItem([title, ''])
        item.setExpanded(True)
        self.tree.addTopLevelItem(item)
        return item

    @staticmethod
    def _add_value_item(
        parent: QTreeWidgetItem,
        label: str,
        value: str,
        *,
        tooltip: str = '',
    ) -> None:
        item = QTreeWidgetItem([label, value])
        if tooltip:
            item.setToolTip(0, tooltip)
            item.setToolTip(1, tooltip)
        parent.addChild(item)

    def _format_field_value(
        self,
        *,
        state_name: str,
        field_name: str,
        value: Any,
        choice_maps: dict[str, dict[str, Any]],
    ) -> str:
        if value is None or value == '':
            return str(self._texts.get('empty_value', '—'))
        if isinstance(value, bool):
            return self._format_bool(value)

        normalized_value = value
        if state_name == 'main' and field_name == 'work_mode':
            normalized_value = normalize_work_mode(value)
        if state_name == 'settings' and field_name == 'multi_gpu_mode':
            normalized_value = normalize_multi_gpu_mode(
                value,
                use_multi_gpu_fallback=bool(getattr(self._settings_state, 'use_multi_gpu', False)),
            )
        if state_name == 'settings' and field_name == 'patch_batch_sync_mode':
            normalized_value = normalize_patch_batch_sync_mode(value)

        if field_name in choice_maps:
            choice_map = choice_maps[field_name]
            mapped = choice_map.get(str(normalized_value))
            if mapped:
                return str(mapped)

        if isinstance(normalized_value, tuple):
            return self._format_tuple(normalized_value)
        if isinstance(normalized_value, float):
            return format(normalized_value, 'g')
        return str(normalized_value)

    def _format_bool(self, value: bool) -> str:
        return str(
            self._texts.get('bool_true', 'Да') if bool(value) else self._texts.get('bool_false', 'Нет')
        )

    def _format_status(self, status: str) -> str:
        status_map = _as_dict(self._texts.get('status_values'))
        return str(status_map.get(status, status))

    def _format_tuple(self, value: tuple[Any, ...]) -> str:
        if len(value) == 2:
            return f'{self._format_scalar(value[0])} x {self._format_scalar(value[1])}'
        return ', '.join(self._format_scalar(part) for part in value)

    def _format_scalar(self, value: Any) -> str:
        if isinstance(value, bool):
            return self._format_bool(value)
        if isinstance(value, float):
            return format(value, 'g')
        if value is None or value == '':
            return str(self._texts.get('empty_value', '—'))
        return str(value)

    def _on_restore_clicked(self) -> None:
        self.restore_requested.emit(
            clone_main_window_state(self._main_window_state),
            replace(self._settings_state),
        )
        self.accept()
