"""Product dialog for adding a two-root image layer."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from kraken_manager.workspace import (
    ImageConversionSettings,
    LayerSourceScan,
    WorkspaceValidationError,
    validate_workspace_name,
)
from kraken_manager.domain.project import LayerType


class _ScanThread(QThread):
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        scanner: Callable[..., LayerSourceScan],
        directory: str,
        maximum_frames: int,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._scanner = scanner
        self._directory = directory
        self._maximum_frames = maximum_frames

    def run(self) -> None:
        try:
            result = self._scanner(self._directory, maximum_frames=self._maximum_frames)
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.succeeded.emit(result)


class _DirectoryField(QWidget):
    changed = pyqtSignal(str)

    def __init__(self, *, title: str, object_name: str, parent=None) -> None:
        super().__init__(parent)
        self.title = title
        self.edit = QLineEdit(self)
        self.edit.setObjectName(object_name)
        self.button = QPushButton("Обзор…", self)
        self.button.setObjectName(f"{object_name}Browse")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.edit, 1)
        layout.addWidget(self.button)
        self.button.clicked.connect(self.choose)
        self.edit.textChanged.connect(self.changed)

    def choose(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            self.title,
            self.edit.text().strip(),
        )
        if directory:
            self.edit.setText(directory)

    def text(self) -> str:
        return self.edit.text().strip()


class LayerCreationDialog(QDialog):
    """Create either a managed-copy or external-directory layer."""

    def __init__(
        self,
        *,
        maximum_frames: int,
        scanner: Callable[..., LayerSourceScan],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.maximum_frames = int(maximum_frames)
        self.scanner = scanner
        self.scan_result: LayerSourceScan | None = None
        self._scan_thread: _ScanThread | None = None
        self.setObjectName("layerCreationDialog")
        self.setWindowTitle("Добавить слой")
        self.setMinimumWidth(720)

        root = QVBoxLayout(self)
        intro = QLabel(
            "Укажите имя и источник слоя. Kraken проверит нумерацию кадров до записи файлов.",
            self,
        )
        intro.setWordWrap(True)
        intro.setObjectName("layerCreationIntro")
        root.addWidget(intro)

        identity = QFormLayout()
        self.name_edit = QLineEdit(self)
        self.name_edit.setObjectName("layerName")
        self.name_edit.setPlaceholderText("Например, Metal 1")
        self.type_combo = QComboBox(self)
        self.type_combo.setObjectName("layerType")
        labels = {
            LayerType.METAL: "Металл",
            LayerType.CONTACT: "Контакты",
            LayerType.GATE: "Затвор",
            LayerType.DIFFUSION: "Диффузия",
        }
        for value in LayerType:
            self.type_combo.addItem(labels[value], value.value)
        identity.addRow("Название слоя *", self.name_edit)
        identity.addRow("Тип", self.type_combo)
        root.addLayout(identity)

        self.tabs = QTabWidget(self)
        self.tabs.setObjectName("layerSourceMode")
        self.disk_tab = QWidget(self.tabs)
        self.manual_tab = QWidget(self.tabs)
        self.tabs.addTab(self.disk_tab, "С диска")
        self.tabs.addTab(self.manual_tab, "Вручную")
        root.addWidget(self.tabs, 1)
        self._build_disk_tab()
        self._build_manual_tab()

        self.validation_label = QLabel(self)
        self.validation_label.setObjectName("layerValidationMessage")
        self.validation_label.setWordWrap(True)
        self.validation_label.setStyleSheet("color:#fca5a5;")
        root.addWidget(self.validation_label)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        self.ok_button = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.cancel_button = self.buttons.button(
            QDialogButtonBox.StandardButton.Cancel
        )
        self.ok_button.setText("Создать слой")
        self.cancel_button.setText("Отмена")
        self.ok_button.setObjectName("createLayerConfirm")
        self.ok_button.setEnabled(False)
        self.buttons.accepted.connect(self._try_accept)
        self.buttons.rejected.connect(self.reject)
        root.addWidget(self.buttons)

        self.name_edit.textChanged.connect(self._sync_validity)
        self.tabs.currentChanged.connect(self._sync_validity)
        self.tabs.currentChanged.connect(self._sync_geometry)
        self.manual_images.changed.connect(self._sync_validity)
        self._sync_validity()
        self._sync_geometry()

    def _build_disk_tab(self) -> None:
        layout = QVBoxLayout(self.disk_tab)
        self.disk_source = _DirectoryField(
            title="Выберите корень импорта",
            object_name="layerDiskSource",
            parent=self.disk_tab,
        )
        layout.addWidget(QLabel("Корень импорта *", self.disk_tab))
        layout.addWidget(self.disk_source)
        scan_row = QHBoxLayout()
        self.scan_button = QPushButton("Сканировать", self.disk_tab)
        self.scan_button.setObjectName("scanLayerSource")
        self.scan_state = QLabel("Выберите папку для анализа.", self.disk_tab)
        self.scan_state.setObjectName("layerScanState")
        scan_row.addWidget(self.scan_button)
        scan_row.addWidget(self.scan_state, 1)
        layout.addLayout(scan_row)
        self.scan_button.clicked.connect(self.start_scan)
        self.disk_source.changed.connect(self._source_changed)

        detection = QGroupBox("Обнаружено", self.disk_tab)
        detection_form = QFormLayout(detection)
        self.working_directory_label = QLabel("—", detection)
        self.working_directory_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.jpg_count_label = QLabel("0", detection)
        self.bmp_count_label = QLabel("0", detection)
        self.ssc_count_label = QLabel("0", detection)
        self.prv_count_label = QLabel("0", detection)
        detection_form.addRow("Рабочая папка", self.working_directory_label)
        detection_form.addRow(
            "Нумерация файлов",
            QLabel(
                f"0…{max(0, self.maximum_frames - 1)} "
                "(последнее целое число в имени)",
                detection,
            ),
        )
        detection_form.addRow("JPG/JPEG", self.jpg_count_label)
        detection_form.addRow("BMP", self.bmp_count_label)
        detection_form.addRow("SSC", self.ssc_count_label)
        detection_form.addRow("PRV", self.prv_count_label)
        layout.addWidget(detection)

        transform = QGroupBox("Подготовка изображений", self.disk_tab)
        transform_layout = QVBoxLayout(transform)
        format_row = QHBoxLayout()
        self.jpg_radio = QRadioButton("Преобразовать BMP в JPG", transform)
        self.png_radio = QRadioButton("Преобразовать BMP в PNG", transform)
        self.jpg_radio.setChecked(True)
        format_row.addWidget(self.jpg_radio)
        format_row.addWidget(self.png_radio)
        format_row.addStretch(1)
        transform_layout.addLayout(format_row)

        self.format_settings = QStackedWidget(transform)
        jpeg = QWidget(self.format_settings)
        jpeg_form = QFormLayout(jpeg)
        self.jpeg_quality = QSpinBox(jpeg)
        self.jpeg_quality.setRange(1, 95)
        self.jpeg_quality.setValue(95)
        self.jpeg_subsampling = QComboBox(jpeg)
        self.jpeg_subsampling.addItems(["4:4:4", "4:2:2", "4:2:0"])
        self.jpeg_optimize = QCheckBox("Оптимизировать кодирование", jpeg)
        self.jpeg_progressive = QCheckBox("Прогрессивный JPEG", jpeg)
        jpeg_form.addRow("Качество", self.jpeg_quality)
        jpeg_form.addRow("Цветовая субдискретизация", self.jpeg_subsampling)
        jpeg_form.addRow(self.jpeg_optimize)
        jpeg_form.addRow(self.jpeg_progressive)
        png = QWidget(self.format_settings)
        png_form = QFormLayout(png)
        self.png_compression = QSpinBox(png)
        self.png_compression.setRange(0, 9)
        self.png_compression.setValue(6)
        self.png_optimize = QCheckBox("Максимально оптимизировать", png)
        png_form.addRow("Уровень сжатия", self.png_compression)
        png_form.addRow(self.png_optimize)
        self.format_settings.addWidget(jpeg)
        self.format_settings.addWidget(png)
        transform_layout.addWidget(self.format_settings)
        self.flip_horizontal = QCheckBox("Отразить по горизонтали", transform)
        self.flip_vertical = QCheckBox("Отразить по вертикали", transform)
        transform_layout.addWidget(self.flip_horizontal)
        transform_layout.addWidget(self.flip_vertical)
        self.bmp_transform_group = transform
        self.bmp_transform_group.setVisible(False)
        layout.addWidget(transform)
        layout.addStretch(1)
        self.jpg_radio.toggled.connect(
            lambda checked: self.format_settings.setCurrentIndex(0 if checked else 1)
        )
        self.png_optimize.toggled.connect(
            lambda checked: self.png_compression.setEnabled(not checked)
        )

    def _build_manual_tab(self) -> None:
        layout = QFormLayout(self.manual_tab)
        hint = QLabel(
            "Kraken запомнит абсолютные пути. Файлы не копируются и внешние папки не удаляются.",
            self.manual_tab,
        )
        hint.setWordWrap(True)
        self.manual_images = _DirectoryField(
            title="Выберите папку изображений",
            object_name="manualImageDirectory",
            parent=self.manual_tab,
        )
        self.manual_ssc = _DirectoryField(
            title="Выберите папку SSC",
            object_name="manualSscDirectory",
            parent=self.manual_tab,
        )
        self.manual_prv = _DirectoryField(
            title="Выберите папку PRV",
            object_name="manualPrvDirectory",
            parent=self.manual_tab,
        )
        layout.addRow(hint)
        layout.addRow("Изображения *", self.manual_images)
        layout.addRow("SSC", self.manual_ssc)
        layout.addRow("PRV", self.manual_prv)

    @property
    def mode(self) -> str:
        return "disk" if self.tabs.currentWidget() is self.disk_tab else "manual"

    @property
    def layer_name(self) -> str:
        return self.name_edit.text().strip()

    @property
    def layer_type(self) -> LayerType:
        return LayerType(str(self.type_combo.currentData()))

    def conversion_settings(self) -> ImageConversionSettings:
        return ImageConversionSettings(
            target_format="jpg" if self.jpg_radio.isChecked() else "png",
            flip_horizontal=self.flip_horizontal.isChecked(),
            flip_vertical=self.flip_vertical.isChecked(),
            jpeg_quality=self.jpeg_quality.value(),
            jpeg_subsampling=self.jpeg_subsampling.currentText(),
            jpeg_optimize=self.jpeg_optimize.isChecked(),
            jpeg_progressive=self.jpeg_progressive.isChecked(),
            png_compression=self.png_compression.value(),
            png_optimize=self.png_optimize.isChecked(),
        )

    def _source_changed(self, _value: str) -> None:
        self.scan_result = None
        self.scan_state.setText("Источник изменён — запустите сканирование.")
        self.working_directory_label.setText("—")
        for label in (
            self.jpg_count_label,
            self.bmp_count_label,
            self.ssc_count_label,
            self.prv_count_label,
        ):
            label.setText("0")
        self._sync_validity()
        self._sync_geometry()

    def start_scan(self) -> None:
        directory = self.disk_source.text()
        if not directory:
            self.validation_label.setText("Выберите корень импорта.")
            return
        if self._scan_thread is not None and self._scan_thread.isRunning():
            return
        self.scan_result = None
        self.scan_button.setEnabled(False)
        self.scan_state.setText("Сканирование…")
        self.validation_label.clear()
        thread = _ScanThread(self.scanner, directory, self.maximum_frames, self)
        thread.succeeded.connect(self._scan_succeeded)
        thread.failed.connect(self._scan_failed)
        thread.finished.connect(lambda: self.scan_button.setEnabled(True))
        self._scan_thread = thread
        thread.start()
        self._sync_validity()
        self._sync_geometry()

    def _scan_succeeded(self, result: LayerSourceScan) -> None:
        self.scan_result = result
        self.working_directory_label.setText(result.working_directory or "Не найдена")
        self.jpg_count_label.setText(str(len(result.jpg_files)))
        self.bmp_count_label.setText(str(len(result.bmp_files)))
        self.ssc_count_label.setText(str(len(result.ssc_files)))
        self.prv_count_label.setText(str(len(result.prv_files)))
        if result.ready:
            self.scan_state.setText(
                f"Готово: {len(result.image_files)} изображений, "
                f"{result.total_files} файлов во всём источнике."
            )
            self.validation_label.clear()
        else:
            self.scan_state.setText("Источник требует исправления.")
            self.validation_label.setText("\n".join(result.issues))
        self.bmp_transform_group.setTitle(
            "Преобразование BMP" if result.bmp_files else "Подготовка JPG"
        )
        self.bmp_transform_group.setVisible(result.ready)
        for widget in (self.jpg_radio, self.png_radio, self.format_settings):
            widget.setVisible(bool(result.bmp_files))
        self._sync_validity()
        self._sync_geometry()

    def _scan_failed(self, message: str) -> None:
        self.scan_result = None
        self.scan_state.setText("Сканирование завершилось ошибкой.")
        self.validation_label.setText(message)
        self._sync_validity()
        self._sync_geometry()

    def _sync_geometry(self, *_args) -> None:
        if self.mode == "manual":
            height = 210
        elif self.scan_result is None:
            height = 300
        elif not self.scan_result.ready:
            height = 300
        elif self.scan_result.bmp_files:
            height = 550
        else:
            height = 420
        self.tabs.setFixedHeight(height)
        self.adjustSize()

    def _sync_validity(self, *_args) -> None:
        name_ready = False
        if self.layer_name:
            try:
                validate_workspace_name(
                    self.layer_name,
                    field_name="Название слоя",
                )
                name_ready = True
                if self.validation_label.text().startswith("Название слоя:"):
                    self.validation_label.clear()
            except WorkspaceValidationError as exc:
                self.validation_label.setText(str(exc))
        source_ready = (
            bool(self.scan_result and self.scan_result.ready)
            if self.mode == "disk"
            else bool(self.manual_images.text())
        )
        self.ok_button.setEnabled(name_ready and source_ready)

    def _try_accept(self) -> None:
        if not self.layer_name:
            self.validation_label.setText("Введите название слоя.")
            self.name_edit.setFocus()
            return
        try:
            validate_workspace_name(
                self.layer_name,
                field_name="Название слоя",
            )
        except WorkspaceValidationError as exc:
            self.validation_label.setText(str(exc))
            self.name_edit.setFocus()
            return
        if self.mode == "disk" and not (self.scan_result and self.scan_result.ready):
            self.validation_label.setText("Сначала выполните успешное сканирование источника.")
            return
        if self.mode == "manual" and not self.manual_images.text():
            self.validation_label.setText("Укажите папку изображений.")
            return
        if self.mode == "manual":
            for label, value, required in (
                ("Изображения", self.manual_images.text(), True),
                ("SSC", self.manual_ssc.text(), False),
                ("PRV", self.manual_prv.text(), False),
            ):
                if not value and not required:
                    continue
                if not Path(value).expanduser().is_dir():
                    self.validation_label.setText(
                        f"{label}: указанная папка недоступна."
                    )
                    return
        self.accept()

    def reject(self) -> None:
        if self._scan_thread is not None and self._scan_thread.isRunning():
            self.validation_label.setText(
                "Дождитесь завершения сканирования перед закрытием окна."
            )
            return
        super().reject()


__all__ = ["LayerCreationDialog"]
