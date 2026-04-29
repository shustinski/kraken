from __future__ import annotations

import shutil
from pathlib import Path

from PyQt6 import QtGui, QtWidgets
from PyQt6.QtCore import QThread
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QInputDialog,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QTextBrowser,
    QListView,
    QTreeView,
)
from kraken_core.theme import add_theme_menu, apply_app_theme, normalize_theme

from csliser import __version__
from csliser.application.use_cases import BuildTransferPlan
from csliser.domain.models import FileOperation, OperationPlan, OperationResult, ProcessingConfig, SelectionMode, SourceFolder
from csliser.domain.planner import PlanningError, discover_extensions
from csliser.domain.ranges import FrameRangeError
from csliser.infrastructure.settings_store import CSliserPreset, WindowSettings, WindowSettingsStore
from csliser.presentation.qt.widgets import AnimatedToggle, ExtensionCheckbox
from csliser.presentation.qt.worker import TransferWorker

_IDLE_TEXT = "Жду начала копирования, чтобы информировать о своей работе"

_LEGACY_DEFAULT_EXTENSIONS = (".jpg", ".bmp", ".cif")


def legacy_source_extension_state(path: Path, *, dynamic_extensions: bool) -> tuple[tuple[str, ...], set[str]]:
    """Return extension list and default checks using the legacy CSliser rules."""
    if dynamic_extensions:
        extensions = discover_extensions(path, defaults=_LEGACY_DEFAULT_EXTENSIONS)
        return extensions, {item for item in extensions if item in _LEGACY_DEFAULT_EXTENSIONS}

    drive = path.drive.upper()
    if not drive:
        drive = str(path).replace("\\", "/").split("/", maxsplit=1)[0].upper()
    active: set[str] = set()
    if drive == "X:":
        active.add(".jpg")
    if drive == "Z:":
        active.add(".cif")
    return _LEGACY_DEFAULT_EXTENSIONS, active


def duplicate_source_folder_names(paths: list[Path]) -> tuple[Path, Path] | None:
    seen: dict[str, Path] = {}
    for path in paths:
        key = path.name.lower()
        if key in seen:
            return seen[key], path
        seen[key] = path
    return None


_BUTTON_STYLE = (
    "QPushButton {background-color:rgb(204, 204, 204); "
    "border-style: outset; border-width: 2px; "
    "border-radius: 10px; border-color: beige; padding: 6px;}"
    "QPushButton:hover {background-color: rgb(204, 239, 255);}"
)
_COPY_BUTTON_STYLE_DISABLED = (
    "QPushButton {background-color:rgb(217,217,217); "
    "border-style: outset; border-width: 2px; "
    "border-radius: 10px; border-color: beige; padding: 6px; color: #555555;}"
    "QPushButton:hover {background-color: rgb(204, 239, 255);}"
)
_COPY_BUTTON_STYLE_ENABLED = (
    "QPushButton {background-color:rgb(153,255,204); "
    "border-style: outset; border-width: 2px; "
    "border-radius: 10px; border-color: beige; padding: 6px; color: #111111;}"
    "QPushButton:hover {background-color: rgb(102, 255, 102);}"
)
_FINISH_BUTTON_STYLE = (
    "QPushButton {background-color:rgb(255, 102, 102); "
    "border-style: outset; border-width: 2px; "
    "border-radius: 10px; border-color: beige; padding: 6px; color: #111111;}"
    "QPushButton:hover {background-color: rgb(255, 51, 51);}"
)


class CSliserWindow(QMainWindow):
    """Original CSliser layout backed by the new testable application layer."""

    def __init__(self, settings_store: WindowSettingsStore | None = None) -> None:
        super().__init__()
        self._settings_store = settings_store or WindowSettingsStore()
        self._source_folders: dict[Path, dict[str, set[str] | tuple[str, ...]]] = {}
        self._source_line_edits: list[QLineEdit] = []
        self._extension_checkboxes: list[ExtensionCheckbox] = []
        self._current_plan: OperationPlan | None = None
        self._worker: TransferWorker | None = None
        self._worker_thread: QThread | None = None
        self._current_operation = FileOperation.COPY
        self._busy = False
        self._theme = "dark"
        self._dynamic_extensions = False

        self.font = QtGui.QFont("sans-serif")
        self.font.setPixelSize(14)
        self.font.setBold(True)

        self._build_ui()
        self._create_menu_bar()
        self._restore_settings()
        self._connect_signals()
        self._apply_theme(self._theme)
        self._refresh_state()

    def closeEvent(self, event) -> None:  # noqa: N802
        self._save_settings()
        super().closeEvent(event)

    def _build_ui(self) -> None:
        self.setWindowTitle("CSlicer")
        self.setGeometry(100, 100, 800, 300)

        self.page = QtWidgets.QWidget(self)
        self.setCentralWidget(self.page)
        self.main_layout = QGridLayout(self.page)
        self.main_widget = QGroupBox()
        self.gridapp = QGridLayout()
        self.main_widget.setLayout(self.gridapp)
        self.main_layout.addWidget(self.main_widget)

        self.gridapp.setColumnStretch(0, 1)
        self.gridapp.setColumnStretch(1, 10)
        self.gridapp.setColumnStretch(2, 2)
        self.gridapp.setColumnStretch(3, 1)

        self.source_select_label = QtWidgets.QLabel("Копировать:")
        self.source_select_label.setFont(self.font)
        self.gridapp.addWidget(self.source_select_label, 0, 0)

        self.copy_dir_groupbox = QGroupBox()
        self.copy_dir_groupbox.setFont(self.font)
        self.copy_groupbox_layout = QGridLayout()
        self.copy_groupbox_layout.setColumnStretch(0, 7)
        self.copy_groupbox_layout.setColumnStretch(1, 1)
        self.copy_dir_groupbox.setLayout(self.copy_groupbox_layout)

        self.plus_dirs_button = QPushButton("+")
        self.plus_dirs_button.setFont(self.font)
        self.plus_dirs_button.setMinimumSize(48, 32)
        self.plus_dirs_button.setStyleSheet(_COPY_BUTTON_STYLE_ENABLED)
        self.copy_groupbox_layout.addWidget(self.plus_dirs_button, 0, 0, 1, 2)
        self.gridapp.addWidget(self.copy_dir_groupbox, 0, 1, 1, 3)

        self.destination_folder_label = QtWidgets.QLabel("Поместить в:")
        self.destination_folder_label.setFont(self.font)
        self.gridapp.addWidget(self.destination_folder_label, 1, 0)

        self.destination_folder_lineedit = QLineEdit()
        self.destination_folder_lineedit.setFont(self.font)
        self.gridapp.addWidget(self.destination_folder_lineedit, 1, 1)

        self.destination_folder_select = QPushButton("Выбрать")
        self.destination_folder_select.setFont(self.font)
        self.destination_folder_select.setStyleSheet(_BUTTON_STYLE)
        self.gridapp.addWidget(self.destination_folder_select, 1, 2, 1, 2)

        self.first_frame_label = QtWidgets.QLabel("Кадры:")
        self.first_frame_label.setFont(self.font)
        self.gridapp.addWidget(self.first_frame_label, 2, 0)

        self.first_frame_lineedit = QLineEdit()
        self.first_frame_lineedit.setFont(self.font)
        self.first_frame_lineedit.setPlaceholderText("Например: 10-20;100:300,500:600;700")
        self.gridapp.addWidget(self.first_frame_lineedit, 2, 1)

        self.copy_groupbox = QGroupBox("Режим копирования")
        self.copy_groupbox.setFont(self.font)
        self.copy_mode_layout = QGridLayout()
        self.copy_all_radiobutton = QRadioButton("Весь диапазон")
        self.copy_area_radiobutton = QRadioButton("Прямоугольная область")
        self.copy_area_radiobutton.setChecked(True)
        self.copy_mode_layout.addWidget(self.copy_all_radiobutton, 0, 0)
        self.copy_mode_layout.addWidget(self.copy_area_radiobutton, 1, 0)
        self.copy_groupbox.setLayout(self.copy_mode_layout)
        self.gridapp.addWidget(self.copy_groupbox, 2, 2, 2, 3)

        self.frames_in_row_label = QtWidgets.QLabel("Кадров в строке:")
        self.frames_in_row_label.setFont(self.font)
        self.gridapp.addWidget(self.frames_in_row_label, 3, 0)

        self.frames_in_row_lineedit = QLineEdit()
        self.frames_in_row_lineedit.setFont(self.font)
        self.gridapp.addWidget(self.frames_in_row_lineedit, 3, 1)

        self.copy_cut_groupbox = QGroupBox()
        self.copy_cut_layout = QGridLayout()
        self.copy_cut_groupbox.setLayout(self.copy_cut_layout)
        self.copy_button = QPushButton("Копировать")
        self.move_button = QPushButton("Переместить")
        self.delete_button = QPushButton("Удалить")
        for column, button in enumerate((self.copy_button, self.move_button, self.delete_button)):
            button.setFont(self.font)
            button.setEnabled(False)
            self.copy_cut_layout.addWidget(button, 0, column)
        self.gridapp.addWidget(self.copy_cut_groupbox, 4, 0, 1, 2)

        self.finish_button = QPushButton("Остановить копирование")
        self.finish_button.setFont(self.font)
        self.finish_button.setStyleSheet(_FINISH_BUTTON_STYLE)
        self.finish_button.setVisible(False)
        self.gridapp.addWidget(self.finish_button, 4, 0, 1, 2)

        self.add_extension_l = QtWidgets.QLabel("Добавить расширение")
        self.add_extension_l.setFont(self.font)
        self.gridapp.addWidget(self.add_extension_l, 4, 2, 1, 1)

        self.add_extension_checkbox = AnimatedToggle()
        self.add_extension_checkbox.setChecked(True)
        self.gridapp.addWidget(self.add_extension_checkbox, 4, 3)

        self.label_info = QtWidgets.QLabel(_IDLE_TEXT)
        self.label_info.setFont(self.font)
        self.gridapp.addWidget(self.label_info, 5, 0, 1, 4)

        self.progress = QProgressBar()
        self.progress.setStyleSheet("text-align: center;")
        self.gridapp.addWidget(self.progress, 6, 0, 1, 3)

        self.settings_btn = QPushButton()
        self.settings_btn.setText("⚙")
        self.settings_btn.setToolTip("Настройки")
        self.settings_btn.setFont(self.font)
        self.gridapp.addWidget(self.settings_btn, 6, 3)

    def _create_menu_bar(self) -> None:
        file_menu = self.menuBar().addMenu("Действия")
        add_theme_menu(self, initial_theme=self._theme, on_theme_changed=self._apply_theme)
        help_menu = self.menuBar().addMenu("Справка")
        self._add_menu_action(file_menu, "Создать полный пресет", lambda: self._save_preset(include_sources=True))
        self._add_menu_action(file_menu, "Создать пресет кадров", lambda: self._save_preset(include_sources=False))
        self._add_menu_action(file_menu, "Восстановить полный пресет", lambda: self._restore_preset(include_sources=True))
        self._add_menu_action(file_menu, "Восстановить пресет кадров", lambda: self._restore_preset(include_sources=False))
        self._add_menu_action(help_menu, "О версии", self._show_about)

    def _add_menu_action(self, menu: QMenu, text: str, callback) -> None:
        action = QtGui.QAction(text, self)
        action.triggered.connect(callback)
        menu.addAction(action)

    def _connect_signals(self) -> None:
        self.copy_button.clicked.connect(lambda: self._start_operation(FileOperation.COPY))
        self.move_button.clicked.connect(lambda: self._start_operation(FileOperation.MOVE))
        self.delete_button.clicked.connect(lambda: self._start_operation(FileOperation.DELETE))
        self.plus_dirs_button.clicked.connect(self._select_source)
        self.destination_folder_select.clicked.connect(self._select_destination)
        self.finish_button.clicked.connect(self._cancel_worker)
        self.copy_all_radiobutton.clicked.connect(self._disable_frames_in_row)
        self.copy_area_radiobutton.clicked.connect(self._disable_frames_in_row)
        self.settings_btn.clicked.connect(self._open_settings_dialog)

        for widget in (self.destination_folder_lineedit, self.frames_in_row_lineedit, self.first_frame_lineedit):
            widget.textChanged.connect(self._input_text_changed)

    def _restore_settings(self) -> None:
        settings = self._settings_store.load()
        self.font.setPixelSize(settings.font_size)
        self._apply_font(self)
        self.destination_folder_lineedit.setText(settings.destination)
        self.first_frame_lineedit.setText(settings.frame_expression)
        self.frames_in_row_lineedit.setText(str(settings.frames_per_row))
        self.add_extension_checkbox.setChecked(settings.add_extension_prefix)
        self._dynamic_extensions = settings.dynamic_extensions
        self.copy_all_radiobutton.setChecked(settings.selection_mode == SelectionMode.FULL_RANGE.value)
        self.copy_area_radiobutton.setChecked(settings.selection_mode != SelectionMode.FULL_RANGE.value)
        self._disable_frames_in_row()

    def _save_settings(self) -> None:
        frames_per_row = int(self.frames_in_row_lineedit.text() or "135")
        self._settings_store.save(
            WindowSettings(
                destination=self.destination_folder_lineedit.text().strip(),
                frame_expression=self.first_frame_lineedit.text().strip(),
                frames_per_row=frames_per_row,
                font_size=self.font.pixelSize(),
                add_extension_prefix=self.add_extension_checkbox.isChecked(),
                selection_mode=self._selection_mode().value,
                operation=self._current_operation.value,
                dynamic_extensions=self._dynamic_extensions,
            )
        )

    def _select_source(self) -> None:
        initial = next(reversed(self._source_folders), Path.home()).parent if self._source_folders else Path.home()
        dialog = self._create_directory_dialog("Выберите папки", initial=initial, multi_select=True)
        if dialog.exec() != QFileDialog.DialogCode.Accepted:
            return
        for folder in dialog.selectedFiles():
            self._add_source_path(Path(folder))
        self._source_ui()
        self._refresh_state()

    def _create_directory_dialog(
        self,
        title: str,
        *,
        initial: str | Path | None = None,
        multi_select: bool = False,
    ) -> QFileDialog:
        dialog = QFileDialog(self, title, str(initial or ""))
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        if multi_select:
            for view_type in (QListView, QTreeView):
                for view in dialog.findChildren(view_type):
                    view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        return dialog

    def _add_source_path(self, path: Path) -> None:
        path = Path(path)
        if path not in self._source_folders:
            extensions, active = legacy_source_extension_state(path, dynamic_extensions=self._dynamic_extensions)
            self._source_folders[path] = {"all_extensions": extensions, "active_extensions": active}

    def _source_ui(self) -> None:
        self._source_line_edits = []
        self._extension_checkboxes = []
        while self.copy_groupbox_layout.count():
            item = self.copy_groupbox_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                if widget is self.plus_dirs_button:
                    widget.setParent(self.copy_dir_groupbox)
                    continue
                widget.deleteLater()

        for row, path in enumerate(list(self._source_folders)):
            folder_edit = QLineEdit(str(path))
            folder_edit.setFont(self.font)
            folder_edit.editingFinished.connect(
                lambda edit=folder_edit, old_path=path: self._source_text_edit_finished(old_path, edit)
            )
            self._source_line_edits.append(folder_edit)
            self.copy_groupbox_layout.addWidget(folder_edit, row, 0)

            extensions_group = QGroupBox()
            extensions_grid = QGridLayout()
            extensions_group.setLayout(extensions_grid)
            self.copy_groupbox_layout.addWidget(extensions_group, row, 1)

            all_extensions = self._source_folders[path]["all_extensions"]
            active_extensions = self._source_folders[path]["active_extensions"]
            assert isinstance(all_extensions, tuple)
            assert isinstance(active_extensions, set)
            for index, extension in enumerate(all_extensions):
                checkbox = ExtensionCheckbox(str(path), extension)
                checkbox.setChecked(extension in active_extensions)
                checkbox.toggled.connect(
                    lambda checked, source=path, ext=extension: self._set_extension(source, ext, checked)
                )
                self._extension_checkboxes.append(checkbox)
                extensions_grid.addWidget(checkbox, index // 4, index % 4)

            delete_folder = QPushButton("-")
            delete_folder.setFont(self.font)
            delete_folder.setMinimumSize(48, 32)
            delete_folder.clicked.connect(lambda _checked=False, source=path: self._delete_folder(source))
            delete_folder.setObjectName("dangerButton")
            delete_folder.setStyleSheet(_FINISH_BUTTON_STYLE)
            self.copy_groupbox_layout.addWidget(delete_folder, row, 2)

        self.copy_groupbox_layout.addWidget(self.plus_dirs_button, len(self._source_folders) + 1, 0, 1, 2)

    def _source_text_edit_finished(self, old_path: Path, edit: QLineEdit) -> None:
        new_path = Path(edit.text())
        if new_path == old_path or old_path not in self._source_folders:
            return
        self._replace_source_path_preserving_order(old_path, new_path)
        self._source_ui()
        self._refresh_state()

    def _set_extension(self, source: Path, extension: str, checked: bool) -> None:
        active_extensions = self._source_folders[source]["active_extensions"]
        assert isinstance(active_extensions, set)
        if checked:
            active_extensions.add(extension)
        else:
            active_extensions.discard(extension)
        self._refresh_state()

    def _delete_folder(self, source: Path) -> None:
        self._source_folders.pop(source, None)
        self._source_ui()
        self._refresh_state()

    def _select_destination(self) -> None:
        last_source_parent = next(reversed(self._source_folders), Path.home()).parent if self._source_folders else Path.home()
        initial = self.destination_folder_lineedit.text().strip() or str(last_source_parent)
        dialog = self._create_directory_dialog("Выберите папку назначения", initial=initial)
        if dialog.exec() != QFileDialog.DialogCode.Accepted:
            return
        selected = dialog.selectedFiles()
        if selected:
            self.destination_folder_lineedit.setText(selected[0])

    def _input_text_changed(self) -> None:
        self._sanitize_numeric_inputs()
        self._current_plan = None
        self._refresh_state()

    def _sanitize_numeric_inputs(self) -> None:
        allowed = set("0123456789-:;,")
        text = self.first_frame_lineedit.text()
        cleaned = "".join(ch for ch in text if ch in allowed)
        if cleaned != text:
            cursor = self.first_frame_lineedit.cursorPosition()
            self.first_frame_lineedit.setText(cleaned)
            self.first_frame_lineedit.setCursorPosition(max(0, cursor - 1))

        row_text = self.frames_in_row_lineedit.text()
        cleaned_row = "".join(ch for ch in row_text if ch.isdigit())
        if cleaned_row != row_text:
            cursor = self.frames_in_row_lineedit.cursorPosition()
            self.frames_in_row_lineedit.setText(cleaned_row)
            self.frames_in_row_lineedit.setCursorPosition(max(0, cursor - 1))

    def _disable_frames_in_row(self) -> None:
        enabled = self.copy_area_radiobutton.isChecked()
        self.frames_in_row_lineedit.setEnabled(enabled)
        self.frames_in_row_label.setEnabled(enabled)
        self._current_plan = None
        self._refresh_state()

    def _selection_mode(self) -> SelectionMode:
        return SelectionMode.FULL_RANGE if self.copy_all_radiobutton.isChecked() else SelectionMode.RECTANGLE

    def _config(self, operation: FileOperation) -> ProcessingConfig:
        self._sync_source_edits()
        sources: list[SourceFolder] = []
        for path, payload in self._source_folders.items():
            active_extensions = payload["active_extensions"]
            assert isinstance(active_extensions, set)
            sources.append(SourceFolder(path=path, extensions=tuple(sorted(active_extensions))))
        destination_text = self.destination_folder_lineedit.text().strip()
        return ProcessingConfig(
            sources=tuple(sources),
            frame_expression=self.first_frame_lineedit.text().strip(),
            selection_mode=self._selection_mode(),
            frames_per_row=int(self.frames_in_row_lineedit.text() or "0"),
            operation=operation,
            destination=Path(destination_text) if destination_text else None,
            add_extension_prefix=self.add_extension_checkbox.isChecked(),
        )

    def _sync_source_edits(self) -> None:
        for edit, old_path in zip(self._source_line_edits, list(self._source_folders), strict=False):
            new_path = Path(edit.text())
            if new_path == old_path or old_path not in self._source_folders:
                continue
            self._replace_source_path_preserving_order(old_path, new_path)

    def _replace_source_path_preserving_order(self, old_path: Path, new_path: Path) -> None:
        rebuilt: dict[Path, dict[str, set[str] | tuple[str, ...]]] = {}
        for path, payload in self._source_folders.items():
            if path == old_path:
                extensions, active = legacy_source_extension_state(
                    new_path,
                    dynamic_extensions=self._dynamic_extensions,
                )
                rebuilt[new_path] = {"all_extensions": extensions, "active_extensions": active}
            else:
                rebuilt[path] = payload
        self._source_folders = rebuilt

    def _start_operation(self, operation: FileOperation) -> None:
        self._current_operation = operation
        self._sync_source_edits()
        if operation != FileOperation.DELETE and not self.add_extension_checkbox.isChecked():
            duplicate = duplicate_source_folder_names(list(self._source_folders))
            if duplicate is not None and not self._confirm_duplicate_source_names(duplicate):
                return
        try:
            plan = BuildTransferPlan().execute(self._config(operation))
        except (FrameRangeError, PlanningError, OSError, ValueError) as exc:
            QMessageBox.warning(self, "Ошибка", str(exc))
            return

        if not plan.operations:
            QMessageBox.information(self, "Информация", "Нет файлов для обработки.")
            return
        if plan.skipped_sources and not self._confirm_skipped_sources(plan):
            return
        if operation in {FileOperation.MOVE, FileOperation.DELETE} and not self._confirm_destructive_operation(operation, plan):
            return
        if operation != FileOperation.DELETE and not self._confirm_destination(plan):
            return

        self._save_settings()
        self._current_plan = plan
        self._start_worker(plan)

    def _confirm_duplicate_source_names(self, duplicate: tuple[Path, Path]) -> bool:
        first, second = duplicate
        answer = QMessageBox.question(
            self,
            "Повторяющиеся папки",
            f"Папки {second} и {first} имеют одно имя, "
            "а добавление расширения выключено. "
            "Файлы окажутся в одной папке. Продолжить?",
        )
        return answer == QMessageBox.StandardButton.Yes

    def _confirm_destination(self, plan: OperationPlan) -> bool:
        destination = plan.config.destination
        if destination is None:
            return False
        if destination.exists() and any(destination.iterdir()):
            answer = QMessageBox.question(
                self,
                "В папке уже есть файлы",
                f"В папке {destination} уже есть файлы. Продолжить?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return False
        disk_target = destination if destination.exists() else next((parent for parent in destination.parents if parent.exists()), Path.cwd())
        free_gib = shutil.disk_usage(disk_target).free / 1024 / 1024 / 1024
        if plan.total_gib > free_gib:
            answer = QMessageBox.question(
                self,
                "Недостаточно места",
                f"Нужно {plan.total_gib:.2f} ГБ, свободно {free_gib:.2f} ГБ. Продолжить?",
            )
            return answer == QMessageBox.StandardButton.Yes
        return True

    def _confirm_skipped_sources(self, plan: OperationPlan) -> bool:
        folders = "\n".join(f"{folder} [{extension}]" for folder, extension in plan.skipped_sources[:20])
        answer = QMessageBox.question(
            self,
            "Пропущенные папки",
            f"В некоторых папках нет файлов нужного формата, они будут пропущены:\n{folders}\n\nПродолжить?",
        )
        return answer == QMessageBox.StandardButton.Yes

    def _confirm_destructive_operation(self, operation: FileOperation, plan: OperationPlan) -> bool:
        verb = "переместить" if operation == FileOperation.MOVE else "удалить"
        answer = QMessageBox.question(
            self,
            "Предупреждение",
            f"Вы собираетесь {verb} {len(plan.operations)} файлов. Продолжить?",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return False
        if operation != FileOperation.DELETE:
            return True
        answer = QMessageBox.question(
            self,
            "Предупреждение",
            "В случае ошибки файлы могут быть утрачены навсегда. Удалить?",
        )
        return answer == QMessageBox.StandardButton.Yes

    def _start_worker(self, plan: OperationPlan) -> None:
        self.progress.setRange(0, len(plan.operations))
        self.progress.setValue(0)
        self._busy = True
        self.copy_cut_groupbox.setVisible(False)
        self.finish_button.setVisible(True)
        self._change_enabling(False)

        self._worker_thread = QThread(self)
        self._worker = TransferWorker(plan)
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress_changed.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker_thread.finished.connect(self._worker_thread.deleteLater)
        self._worker_thread.start()

    def _cancel_worker(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.label_info.setText("Останавливаю после текущего файла...")

    def _on_progress(self, current: int, total: int, path: str) -> None:
        self.progress.setRange(0, total)
        self.progress.setValue(current)
        self.label_info.setText(path)

    def _on_finished(self, result: OperationResult) -> None:
        self._worker = None
        self._worker_thread = None
        self._busy = False
        self.copy_cut_groupbox.setVisible(True)
        self.finish_button.setVisible(False)
        self._change_enabling(True)
        self.progress.setValue(0)
        self.label_info.setText(_IDLE_TEXT)

        if result.errors:
            details = "\n".join(f"{error.source}: {error.message}" for error in result.errors[:100])
            QMessageBox.warning(self, "Пропущенные файлы", details)
        elif result.cancelled:
            QMessageBox.information(self, "Информация", "Операция остановлена.")
        else:
            messages = {
                FileOperation.COPY: "Копирование завершено",
                FileOperation.MOVE: "Перемещение завершено",
                FileOperation.DELETE: "Удаление завершено",
            }
            QMessageBox.information(self, "Информация", messages[self._current_operation])

    def _change_enabling(self, state: bool) -> None:
        for widget in (
            self.source_select_label,
            self.copy_dir_groupbox,
            self.destination_folder_label,
            self.destination_folder_lineedit,
            self.destination_folder_select,
            self.first_frame_label,
            self.first_frame_lineedit,
            self.frames_in_row_label,
            self.frames_in_row_lineedit,
            self.copy_groupbox,
            self.add_extension_l,
            self.add_extension_checkbox,
            self.settings_btn,
        ):
            widget.setEnabled(state)
        if state:
            self._refresh_state()
        else:
            self.copy_button.setEnabled(False)
            self.move_button.setEnabled(False)
            self.delete_button.setEnabled(False)
            self._apply_operation_button_styles()

    def _refresh_state(self) -> None:
        if self._busy:
            self.copy_button.setEnabled(False)
            self.move_button.setEnabled(False)
            self.delete_button.setEnabled(False)
            self._apply_operation_button_styles()
            return
        has_sources = bool(self._source_folders)
        has_frames = bool(self.first_frame_lineedit.text().strip())
        has_destination = bool(self.destination_folder_lineedit.text().strip())
        has_row_count = self.copy_all_radiobutton.isChecked() or self.frames_in_row_lineedit.text().isdigit()
        can_delete = has_sources and has_frames and has_row_count
        can_copy_or_move = can_delete and has_destination
        self.delete_button.setEnabled(can_delete)
        self.copy_button.setEnabled(can_copy_or_move)
        self.move_button.setEnabled(can_copy_or_move)
        self._apply_operation_button_styles()

    def _apply_operation_button_styles(self) -> None:
        self.copy_button.setStyleSheet(
            _COPY_BUTTON_STYLE_ENABLED if self.copy_button.isEnabled() else _COPY_BUTTON_STYLE_DISABLED
        )
        self.move_button.setStyleSheet(_BUTTON_STYLE if self.move_button.isEnabled() else _COPY_BUTTON_STYLE_DISABLED)
        self.delete_button.setStyleSheet(
            _FINISH_BUTTON_STYLE if self.delete_button.isEnabled() else _COPY_BUTTON_STYLE_DISABLED
        )
        self.plus_dirs_button.setStyleSheet(_COPY_BUTTON_STYLE_ENABLED)

    def _open_settings_dialog(self) -> None:
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Настройки")
        layout = QGridLayout(dialog)

        font_label = QtWidgets.QLabel("Размер шрифта:")
        font_edit = QLineEdit(str(self.font.pixelSize()))
        dynamic_checkbox = QtWidgets.QCheckBox("Динамические расширения")
        dynamic_checkbox.setChecked(self._dynamic_extensions)

        file_browser_groupbox = QGroupBox("Файловый браузер")
        file_browser_layout = QGridLayout(file_browser_groupbox)
        qt_file_browser = QRadioButton("QFileDialog")
        qt_file_browser.setChecked(True)
        qt_file_browser.setEnabled(False)
        file_browser_layout.addWidget(qt_file_browser, 0, 0)

        ok_button = QPushButton("Ок")
        apply_button = QPushButton("Применить")

        layout.addWidget(font_label, 0, 0, 1, 2)
        layout.addWidget(font_edit, 0, 2)
        layout.addWidget(dynamic_checkbox, 1, 0, 1, 3)
        layout.addWidget(file_browser_groupbox, 2, 0, 1, 3)
        layout.addWidget(ok_button, 3, 1)
        layout.addWidget(apply_button, 3, 2)

        def apply_settings(close_dialog: bool) -> None:
            if not font_edit.text().isdigit():
                QMessageBox.warning(dialog, "Ошибка", "Размер шрифта должен быть числом.")
                return
            value = int(font_edit.text())
            if value < 8 or value > 72:
                QMessageBox.warning(dialog, "Ошибка", "Размер шрифта должен быть от 8 до 72.")
                return
            self.font.setPixelSize(value)
            self._dynamic_extensions = dynamic_checkbox.isChecked()
            self._apply_font(self)
            self._apply_theme(self._theme)
            self._save_settings()
            if close_dialog:
                dialog.accept()

        ok_button.clicked.connect(lambda: apply_settings(True))
        apply_button.clicked.connect(lambda: apply_settings(False))
        self._apply_font(dialog)
        dialog.exec()

    def _apply_font(self, widget) -> None:
        widget.setFont(self.font)
        for child in widget.findChildren(QtWidgets.QWidget):
            child.setFont(self.font)

    def _save_preset(self, *, include_sources: bool) -> None:
        name, ok = QInputDialog.getText(self, "Имя пресета", "Имя пресета:")
        if not ok or not name.strip():
            return
        self._sync_source_edits()
        preset = CSliserPreset(
            sources=tuple(str(path) for path in self._source_folders),
            destination=self.destination_folder_lineedit.text(),
            frames=self.first_frame_lineedit.text(),
            row_frames=self.frames_in_row_lineedit.text(),
        )
        self._settings_store.save_preset(name.strip(), preset, include_sources=include_sources)

    def _restore_preset(self, *, include_sources: bool) -> None:
        presets = self._settings_store.load_presets(include_sources=include_sources)
        if not presets:
            QMessageBox.warning(self, "Ошибка", "Нет пресетов для восстановления")
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Полное восстановление" if include_sources else "Восстановление кадров")
        layout = QGridLayout(dialog)

        def clear_layout() -> None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()

        def restore_name(name: str) -> None:
            answer = QMessageBox.question(dialog, "Восстановление", f"Восстановить {name}?")
            if answer != QMessageBox.StandardButton.Yes:
                return
            self._apply_preset(presets[name], include_sources=include_sources)
            dialog.accept()

        def delete_name(name: str) -> None:
            answer = QMessageBox.question(dialog, "Удалить", f"Удалить {name}?")
            if answer != QMessageBox.StandardButton.Yes:
                return
            self._settings_store.delete_preset(name, include_sources=include_sources)
            presets.pop(name, None)
            if not presets:
                dialog.accept()
                return
            render_presets()

        def render_presets() -> None:
            clear_layout()
            for row, name in enumerate(presets):
                restore_button = QPushButton(name)
                restore_button.setFont(self.font)
                restore_button.clicked.connect(lambda _checked=False, preset_name=name: restore_name(preset_name))
                layout.addWidget(restore_button, row, 0, 1, 2)

                delete_button = QPushButton("-")
                delete_button.setFont(self.font)
                delete_button.setMinimumSize(48, 32)
                delete_button.setStyleSheet(_FINISH_BUTTON_STYLE)
                delete_button.clicked.connect(lambda _checked=False, preset_name=name: delete_name(preset_name))
                layout.addWidget(delete_button, row, 2)

        render_presets()
        dialog.exec()

    def _apply_preset(self, preset: CSliserPreset, *, include_sources: bool) -> None:
        if include_sources:
            self._source_folders = {}
            for source in preset.sources:
                self._add_source_path(Path(source))
            self.destination_folder_lineedit.setText(preset.destination)
            self._source_ui()
        self.first_frame_lineedit.setText(preset.frames)
        self.frames_in_row_lineedit.setText(preset.row_frames)
        self._refresh_state()

    def _show_about(self) -> None:
        dialog = QMessageBox(self)
        dialog.setWindowTitle("О версии")
        dialog.setText(f"CSlicer\nВерсия: {__version__}")
        details = QTextBrowser()
        details.setPlainText("Kraken plugin build. Интерфейс сохранен в стиле исходного CSlicer.")
        dialog.exec()

    def _apply_theme(self, theme: str) -> None:
        self._theme = normalize_theme(theme)
        apply_app_theme(self._theme)
        if self._theme == "light":
            self.setStyleSheet(f"QWidget {{ font-size: {self.font.pixelSize()}px; }}")
            return
        font_size = self.font.pixelSize()
        stylesheet = """
            QMainWindow, QWidget, QGroupBox {
                background-color: #20242b;
                color: #e7eaf0;
                font-size: __FONT_SIZE__px;
            }
            QGroupBox {
                border: 1px solid #3b4350;
                border-radius: 8px;
                margin-top: 8px;
                padding: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QLineEdit {
                background-color: #11151b;
                color: #f2f4f8;
                border: 2px solid #4a5362;
                border-radius: 6px;
                padding: 5px;
            }
            QPushButton {
                background-color: #343c48;
                color: #f2f4f8;
                border: 1px solid #566173;
                border-radius: 8px;
                padding: 6px;
            }
            QPushButton:hover {
                background-color: #445063;
            }
            QPushButton:disabled {
                background-color: #2b3038;
                color: #7e8795;
            }
            QPushButton#dangerButton {
                background-color: #7d2e33;
            }
            QProgressBar {
                background-color: #11151b;
                color: #f2f4f8;
                border: 1px solid #4a5362;
                border-radius: 6px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #3399ff;
                border-radius: 5px;
            }
            QMenuBar, QMenu {
                background-color: #20242b;
                color: #e7eaf0;
            }
            QMenuBar::item:selected, QMenu::item:selected {
                background-color: #445063;
            }
            """
        self.setStyleSheet(stylesheet.replace("__FONT_SIZE__", str(font_size)))


def run_window() -> int:
    app = QApplication.instance() or QApplication([])
    assert isinstance(app, QApplication)
    window = CSliserWindow()
    window.show()
    return app.exec()
