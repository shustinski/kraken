from __future__ import annotations

from pathlib import Path

from PyQt6 import QtGui
from PyQt6.QtCore import QThread
from PyQt6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QWidget,
)

from kategb.application.dto import AnalyzeVerificationRequest, GenerateManifestRequest, SampleRequest
from kategb.application.use_cases import AnalyzeVerification, BuildSample, GenerateVerificationManifest
from kategb.domain.models import CopyPlan, CopySource, CrystalInfo, LayerInfo
from kategb.infrastructure.file_copy import CopyReport
from kategb.infrastructure.markup_reader import OpenPyxlCrystalInfoReader
from kategb.infrastructure.xml_repository import IncorrectXmlReader
from kategb.presentation.qt.worker import CopyWorker

_HELP_TEXT = (
    "1. Load an Excel markup file or enter layer data manually.\n"
    "2. Select authors and generate frames for verification.\n"
    "3. Save the encrypted manifest and copy source CIF/JPG files.\n"
    "4. After checking, load the manifest and XML result to calculate author quality."
)


class KateGBWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self._crystal_reader = OpenPyxlCrystalInfoReader()
        self._build_sample = BuildSample()
        self._save_manifest = GenerateVerificationManifest()
        self._analyze = AnalyzeVerification(self._crystal_reader, IncorrectXmlReader())
        self._crystal_info: CrystalInfo | None = None
        self._generated_frames: tuple[int, ...] = ()
        self._copy_thread: QThread | None = None
        self._copy_worker: CopyWorker | None = None

        self._font = QtGui.QFont("sans-serif")
        self._font.setPixelSize(14)
        self._font.setBold(True)
        self._build_ui()
        self._connect_signals()
        self._load_demo_layer()

    def _build_ui(self) -> None:
        self.setWindowTitle("KateGB")
        self.resize(980, 680)
        page = QWidget(self)
        self.setCentralWidget(page)
        layout = QGridLayout(page)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        self.tabs.addTab(self._build_sample_tab(), "Sample")
        self.tabs.addTab(self._build_verification_tab(), "Verification")

    def _build_sample_tab(self) -> QWidget:
        page = QWidget()
        grid = QGridLayout(page)
        grid.setColumnStretch(1, 1)

        source_box = QGroupBox("Source data")
        source_layout = QGridLayout(source_box)
        self.markup_path = QLineEdit()
        self.markup_path.setPlaceholderText("Excel markup file (.xlsx)")
        self.load_markup_button = QPushButton("Load markup")
        self.layer_name = QLineEdit()
        self.layer_name.setPlaceholderText("Layer name")
        self.authors = QLineEdit()
        self.authors.setPlaceholderText("Authors separated by comma")
        self.cif_folder = QLineEdit()
        self.cif_folder.setPlaceholderText("CIF source folder")
        self.jpg_folder = QLineEdit()
        self.jpg_folder.setPlaceholderText("Optional JPG source folder")
        for row, (label, widget) in enumerate(
            (
                ("Markup", self.markup_path),
                ("Layer", self.layer_name),
                ("Authors", self.authors),
                ("CIF folder", self.cif_folder),
                ("JPG folder", self.jpg_folder),
            )
        ):
            source_layout.addWidget(QLabel(label), row, 0)
            source_layout.addWidget(widget, row, 1)
        source_layout.addWidget(self.load_markup_button, 0, 2)
        grid.addWidget(source_box, 0, 0, 1, 2)

        settings_box = QGroupBox("Sample settings")
        settings_layout = QGridLayout(settings_box)
        self.percent = QSpinBox()
        self.percent.setRange(1, 100)
        self.percent.setValue(100)
        self.percent.setToolTip("How many frames to take from each selected author.")
        self.frame_range = QLineEdit()
        self.frame_range.setPlaceholderText("Optional, for example 10-250")
        self.frames_in_layer = QSpinBox()
        self.frames_in_layer.setRange(1, 1_000_000)
        self.frames_in_layer.setValue(1)
        self.frames_in_row = QSpinBox()
        self.frames_in_row.setRange(1, 100_000)
        self.frames_in_row.setValue(135)
        self.mode_all = QRadioButton("Whole range")
        self.mode_area = QRadioButton("Rectangular area")
        self.mode_all.setChecked(True)
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(self.mode_all)
        mode_layout.addWidget(self.mode_area)
        for row, (label, widget) in enumerate(
            (
                ("Percent per author", self.percent),
                ("Frame range", self.frame_range),
                ("Frames in layer", self.frames_in_layer),
                ("Frames in row", self.frames_in_row),
            )
        ):
            settings_layout.addWidget(QLabel(label), row, 0)
            settings_layout.addWidget(widget, row, 1)
        settings_layout.addWidget(QLabel("Range mode"), 4, 0)
        settings_layout.addLayout(mode_layout, 4, 1)
        grid.addWidget(settings_box, 1, 0, 1, 2)

        output_box = QGroupBox("Manifest and copy")
        output_layout = QGridLayout(output_box)
        self.check_name = QLineEdit("check")
        self.encryption_key = QLineEdit()
        self.encryption_key.setPlaceholderText("Required encryption key")
        self.output_folder = QLineEdit()
        self.output_folder.setPlaceholderText("Folder for encrypted manifest and copied files")
        self.select_output_button = QPushButton("Select")
        self.generate_button = QPushButton("Generate frames")
        self.save_manifest_button = QPushButton("Save manifest")
        self.copy_button = QPushButton("Copy selected files")
        self.copy_button.setEnabled(False)
        for row, (label, widget) in enumerate(
            (("Check name", self.check_name), ("Key", self.encryption_key), ("Output", self.output_folder))
        ):
            output_layout.addWidget(QLabel(label), row, 0)
            output_layout.addWidget(widget, row, 1)
        output_layout.addWidget(self.select_output_button, 2, 2)
        output_layout.addWidget(self.generate_button, 3, 0)
        output_layout.addWidget(self.save_manifest_button, 3, 1)
        output_layout.addWidget(self.copy_button, 3, 2)
        grid.addWidget(output_box, 2, 0, 1, 2)

        self.frames_preview = QPlainTextEdit()
        self.frames_preview.setReadOnly(True)
        self.frames_preview.setPlaceholderText(_HELP_TEXT)
        self.progress = QProgressBar()
        self.status = QLabel("Ready.")
        grid.addWidget(self.frames_preview, 3, 0, 1, 2)
        grid.addWidget(self.status, 4, 0, 1, 2)
        grid.addWidget(self.progress, 5, 0, 1, 2)
        return page

    def _build_verification_tab(self) -> QWidget:
        page = QWidget()
        grid = QGridLayout(page)
        grid.setColumnStretch(1, 1)
        self.verify_manifest = QLineEdit()
        self.verify_xml = QLineEdit()
        self.verify_key = QLineEdit()
        self.verify_markup = QLineEdit()
        self.select_manifest_button = QPushButton("Select")
        self.select_xml_button = QPushButton("Select")
        self.select_verify_markup_button = QPushButton("Select")
        self.analyze_button = QPushButton("Analyze")
        for row, (label, widget, button) in enumerate(
            (
                ("Manifest", self.verify_manifest, self.select_manifest_button),
                ("Result XML", self.verify_xml, self.select_xml_button),
                ("Key", self.verify_key, None),
                ("Markup", self.verify_markup, self.select_verify_markup_button),
            )
        ):
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(widget, row, 1)
            if button is not None:
                grid.addWidget(button, row, 2)
        grid.addWidget(self.analyze_button, 4, 0, 1, 3)
        self.results_table = QTableWidget(0, 4)
        self.results_table.setHorizontalHeaderLabels(("Author", "Checked", "Incorrect", "Incorrect %"))
        grid.addWidget(self.results_table, 5, 0, 1, 3)
        return page

    def _connect_signals(self) -> None:
        self.load_markup_button.clicked.connect(self._load_markup)
        self.select_output_button.clicked.connect(lambda: self._select_folder(self.output_folder))
        self.generate_button.clicked.connect(self._generate_frames)
        self.save_manifest_button.clicked.connect(self._save_manifest_file)
        self.copy_button.clicked.connect(self._copy_files)
        self.select_manifest_button.clicked.connect(lambda: self._select_file(self.verify_manifest, "Manifest (*.txt)"))
        self.select_xml_button.clicked.connect(lambda: self._select_file(self.verify_xml, "XML (*.xml)"))
        self.select_verify_markup_button.clicked.connect(lambda: self._select_file(self.verify_markup, "Excel (*.xlsx *.xls)"))
        self.analyze_button.clicked.connect(self._analyze_results)

    def _load_demo_layer(self) -> None:
        layer = LayerInfo(
            name="Layer 1",
            author_frames={"Author A": tuple(range(1, 11)), "Author B": tuple(range(11, 21))},
            frames_in_layer=20,
            frames_in_row=10,
        )
        self._crystal_info = CrystalInfo(layers={layer.name: layer})
        self._apply_layer(layer)

    def _load_markup(self) -> None:
        path = self._select_file(self.markup_path, "Excel (*.xlsx *.xls)")
        if not path:
            return
        try:
            self._crystal_info = self._crystal_reader.read(Path(path))
            first_layer = next(iter(self._crystal_info.layers.values()))
            self._apply_layer(first_layer)
            self.status.setText(f"Loaded {len(self._crystal_info.layers)} layer(s).")
        except Exception as exc:
            self._show_error(str(exc))

    def _apply_layer(self, layer: LayerInfo) -> None:
        self.layer_name.setText(layer.name)
        self.authors.setText(", ".join(layer.author_frames))
        self.frames_in_layer.setValue(max(1, layer.frames_in_layer))
        self.frames_in_row.setValue(max(1, layer.frames_in_row))
        if layer.cif_folder:
            self.cif_folder.setText(str(layer.cif_folder))
        if layer.jpg_folder:
            self.jpg_folder.setText(str(layer.jpg_folder))

    def _generate_frames(self) -> None:
        try:
            crystal = self._current_crystal()
            layer_name = self.layer_name.text().strip()
            request = SampleRequest(
                layer_name=layer_name,
                authors=self._selected_authors(),
                percent_per_author=self.percent.value(),
                frame_range_text=self.frame_range.text().strip(),
                selection_mode="all" if self.mode_all.isChecked() else "area",
            )
            self._generated_frames = self._build_sample.execute(crystal, request)
            self.frames_preview.setPlainText("; ".join(str(frame) for frame in self._generated_frames))
            self.copy_button.setEnabled(bool(self._generated_frames))
            self.status.setText(f"Generated {len(self._generated_frames)} frame(s).")
        except Exception as exc:
            self._show_error(str(exc))

    def _save_manifest_file(self) -> None:
        try:
            path = self._save_manifest.execute(
                GenerateManifestRequest(
                    vector_folder=self.cif_folder.text().strip(),
                    layer_name=self.layer_name.text().strip(),
                    check_name=self.check_name.text().strip() or "check",
                    frame_range_text=self.frame_range.text().strip(),
                    selection_mode="all" if self.mode_all.isChecked() else "area",
                    frames=self._generated_frames,
                    encryption_key=self.encryption_key.text(),
                    output_folder=Path(self.output_folder.text()),
                )
            )
            self.status.setText(f"Saved manifest: {path}")
        except Exception as exc:
            self._show_error(str(exc))

    def _copy_files(self) -> None:
        try:
            sources = [CopySource(Path(self.cif_folder.text()), "cif")]
            if self.jpg_folder.text().strip():
                sources.append(CopySource(Path(self.jpg_folder.text()), "jpg"))
            plan = CopyPlan(
                sources=tuple(sources),
                frames=self._generated_frames,
                destination=Path(self.output_folder.text()),
                check_name=self.check_name.text().strip() or "check",
            )
            self._copy_thread = QThread(self)
            self._copy_worker = CopyWorker(plan)
            self._copy_worker.moveToThread(self._copy_thread)
            self._copy_thread.started.connect(self._copy_worker.run)
            self._copy_worker.progress.connect(self._copy_progress)
            self._copy_worker.finished.connect(self._copy_finished)
            self._copy_worker.failed.connect(self._copy_failed)
            self._copy_worker.finished.connect(self._copy_thread.quit)
            self._copy_worker.failed.connect(self._copy_thread.quit)
            self._copy_thread.start()
        except Exception as exc:
            self._show_error(str(exc))

    def _copy_progress(self, done: int, total: int, path: str) -> None:
        self.progress.setMaximum(total)
        self.progress.setValue(done)
        self.status.setText(f"Copying: {path}")

    def _copy_finished(self, report: CopyReport) -> None:
        self.status.setText(f"Copied {len(report.copied_files)} file(s), missing frames: {list(report.missing_frames)}")

    def _copy_failed(self, message: str) -> None:
        self._show_error(message)

    def _analyze_results(self) -> None:
        try:
            results = self._analyze.execute(
                AnalyzeVerificationRequest(
                    manifest_path=Path(self.verify_manifest.text()),
                    encryption_key=self.verify_key.text(),
                    incorrect_xml_path=Path(self.verify_xml.text()),
                    markup_path=Path(self.verify_markup.text()),
                )
            )
            self.results_table.setRowCount(len(results))
            for row, result in enumerate(results):
                values = (
                    result.author,
                    ", ".join(str(frame) for frame in result.checked_frames),
                    ", ".join(str(frame) for frame in result.incorrect_frames),
                    str(result.incorrect_percent),
                )
                for column, value in enumerate(values):
                    self.results_table.setItem(row, column, QTableWidgetItem(value))
        except Exception as exc:
            self._show_error(str(exc))

    def _current_crystal(self) -> CrystalInfo:
        layer_name = self.layer_name.text().strip()
        if self._crystal_info and layer_name in self._crystal_info.layers:
            return self._crystal_info
        author_frames = self._manual_author_frames()
        layer = LayerInfo(
            name=layer_name or "Layer 1",
            author_frames=author_frames,
            frames_in_layer=self.frames_in_layer.value(),
            frames_in_row=self.frames_in_row.value(),
            cif_folder=Path(self.cif_folder.text()) if self.cif_folder.text().strip() else None,
            jpg_folder=Path(self.jpg_folder.text()) if self.jpg_folder.text().strip() else None,
        )
        return CrystalInfo(layers={layer.name: layer})

    def _manual_author_frames(self) -> dict[str, tuple[int, ...]]:
        authors = self._selected_authors()
        if not authors:
            raise ValueError("Enter at least one author.")
        frame_count = self.frames_in_layer.value()
        chunk = max(1, frame_count // len(authors))
        result: dict[str, tuple[int, ...]] = {}
        start = 1
        for index, author in enumerate(authors):
            end = frame_count if index == len(authors) - 1 else min(frame_count, start + chunk - 1)
            result[author] = tuple(range(start, end + 1))
            start = end + 1
        return result

    def _selected_authors(self) -> tuple[str, ...]:
        return tuple(author.strip() for author in self.authors.text().split(",") if author.strip())

    def _select_file(self, line_edit: QLineEdit, file_filter: str) -> str:
        path, _ = QFileDialog.getOpenFileName(self, "Select file", line_edit.text(), file_filter)
        if path:
            line_edit.setText(path)
        return path

    def _select_folder(self, line_edit: QLineEdit) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select folder", line_edit.text())
        if path:
            line_edit.setText(path)

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, "KateGB", message)
        self.status.setText(message)
