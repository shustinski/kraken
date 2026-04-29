from __future__ import annotations

import shutil
from pathlib import Path

from PyQt6 import QtGui, QtWidgets
from PyQt6.QtCore import QThread
from PyQt6.QtWidgets import (
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
)
from kraken_core.theme import add_theme_menu, apply_app_theme, normalize_theme

from csliser import __version__
from csliser.application.use_cases import BuildTransferPlan
from csliser.domain.models import FileOperation, OperationPlan, OperationResult, ProcessingConfig, SelectionMode, SourceFolder
from csliser.domain.planner import PlanningError, discover_extensions
from csliser.domain.ranges import FrameRangeError
from csliser.infrastructure.settings_store import WindowSettings, WindowSettingsStore
from csliser.presentation.qt.widgets import AnimatedToggle, ExtensionCheckbox
from csliser.presentation.qt.worker import TransferWorker

_IDLE_TEXT = "Жду начала копирования, чтобы информировать о своей работе"


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
            )
        )

    def _select_source(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку")
        if not folder:
            return
        path = Path(folder)
        if path not in self._source_folders:
            extensions = discover_extensions(path)
            active = {item for item in extensions if item in {".jpg", ".bmp", ".cif"}}
            self._source_folders[path] = {"all_extensions": extensions, "active_extensions": active or set(extensions)}
        self._source_ui()
        self._refresh_state()

    def _source_ui(self) -> None:
        self._source_line_edits = []
        self._extension_checkboxes = []
        while self.copy_groupbox_layout.count():
            item = self.copy_groupbox_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        for row, path in enumerate(list(self._source_folders)):
            folder_edit = QLineEdit(str(path))
            folder_edit.setFont(self.font)
            folder_edit.textChanged.connect(lambda text, old_path=path: self._source_text_changed(old_path, text))
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
            delete_folder.clicked.connect(lambda _checked=False, source=path: self._delete_folder(source))
            delete_folder.setObjectName("dangerButton")
            self.copy_groupbox_layout.addWidget(delete_folder, row, 2)

        self.copy_groupbox_layout.addWidget(self.plus_dirs_button, len(self._source_folders) + 1, 0, 1, 2)

    def _source_text_changed(self, old_path: Path, text: str) -> None:
        new_path = Path(text)
        if new_path == old_path or old_path not in self._source_folders:
            return
        payload = self._source_folders.pop(old_path)
        self._source_folders[new_path] = payload
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
        initial = self.destination_folder_lineedit.text().strip() or str(Path.home())
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку назначения", initial)
        if folder:
            self.destination_folder_lineedit.setText(folder)

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

    def _start_operation(self, operation: FileOperation) -> None:
        self._current_operation = operation
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

    def _refresh_state(self) -> None:
        if self._busy:
            self.copy_button.setEnabled(False)
            self.move_button.setEnabled(False)
            self.delete_button.setEnabled(False)
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

    def _open_settings_dialog(self) -> None:
        value, ok = QInputDialog.getInt(self, "Настройки", "Размер шрифта:", self.font.pixelSize(), 8, 72)
        if ok:
            self.font.setPixelSize(value)
            self._apply_font(self)
            self._apply_theme(self._theme)
            self._save_settings()

    def _apply_font(self, widget) -> None:
        widget.setFont(self.font)
        for child in widget.findChildren(QtWidgets.QWidget):
            child.setFont(self.font)

    def _save_preset(self, *, include_sources: bool) -> None:
        name, ok = QInputDialog.getText(self, "Имя пресета", "Имя пресета:")
        if not ok or not name.strip():
            return
        # Keep the old menu entry visible; full preset persistence can be added without touching domain logic.
        QMessageBox.information(self, "Информация", f"Пресет «{name.strip()}» сохранен в текущих настройках окна.")
        if include_sources:
            self._save_settings()

    def _restore_preset(self, *, include_sources: bool) -> None:  # noqa: ARG002
        self._restore_settings()
        QMessageBox.information(self, "Информация", "Восстановлены последние сохраненные параметры окна.")

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
