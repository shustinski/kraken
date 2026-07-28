"""Render product-review screenshots for the two-root workspace workflow."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QInputDialog,
    QLineEdit,
    QMessageBox,
    QProgressDialog,
)

from kraken_core.styles import load_shared_stylesheet
from kraken_manager.workspace import scan_layer_source
from kraken_manager.presentation.qt import (
    LayerCreationDialog,
    LayerManagerDialog,
    LayerPipelineSnapshot,
    PipelineLane,
    PipelineNode,
)
from kraken_manager.presentation.qt.models import LayerListItem


OUTPUT = ROOT / "artifacts" / "workspace-ui"


def _capture(app: QApplication, widget, name: str) -> None:
    widget.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, False)
    widget.show()
    widget.adjustSize()
    app.processEvents()
    widget.grab().save(str(OUTPUT / f"{name}.png"))
    widget.close()
    app.processEvents()


def _layer_dialog(maximum_frames: int = 16) -> LayerCreationDialog:
    return LayerCreationDialog(
        maximum_frames=maximum_frames,
        scanner=scan_layer_source,
    )


def main() -> int:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(load_shared_stylesheet())

    with tempfile.TemporaryDirectory() as temporary:
        fixture = Path(temporary)
        jpg = fixture / "jpg"
        bmp = fixture / "bmp"
        mixed = fixture / "mixed"
        manual_ssc = fixture / "ssc"
        for directory in (jpg, bmp, mixed, manual_ssc):
            directory.mkdir()
        for index in (0, 1, 3):
            Image.new("RGB", (20, 14), (40 * index, 90, 140)).save(
                jpg / f"frame_{index}.jpg"
            )
        (jpg / "layout.ssc").write_text("ssc", encoding="utf-8")
        (jpg / "preview.prv").write_text("prv", encoding="utf-8")
        for index in (0, 1):
            Image.new("RGB", (20, 14), (120, 35 * index, 70)).save(
                bmp / f"frame_{index}.bmp"
            )
        Image.new("RGB", (20, 14), "white").save(mixed / "frame_0.jpg")
        Image.new("RGB", (20, 14), "black").save(mixed / "frame_1.bmp")

        manual = _layer_dialog()
        manual.name_edit.setText("Metal 1")
        manual.tabs.setCurrentWidget(manual.manual_tab)
        manual.manual_images.edit.setText(str(jpg))
        manual.manual_ssc.edit.setText(str(manual_ssc))
        _capture(app, manual, "01-manual")

        jpg_dialog = _layer_dialog()
        jpg_dialog.name_edit.setText("Metal JPG")
        jpg_dialog.disk_source.edit.setText(str(jpg))
        jpg_dialog._scan_succeeded(scan_layer_source(jpg, maximum_frames=16))
        _capture(app, jpg_dialog, "02-jpg")

        bmp_jpg = _layer_dialog()
        bmp_jpg.name_edit.setText("Metal BMP JPG")
        bmp_jpg.disk_source.edit.setText(str(bmp))
        bmp_jpg._scan_succeeded(scan_layer_source(bmp, maximum_frames=16))
        _capture(app, bmp_jpg, "03-bmp-to-jpg")

        bmp_png = _layer_dialog()
        bmp_png.name_edit.setText("Metal BMP PNG")
        bmp_png.disk_source.edit.setText(str(bmp))
        bmp_png._scan_succeeded(scan_layer_source(bmp, maximum_frames=16))
        bmp_png.png_radio.setChecked(True)
        bmp_png.flip_vertical.setChecked(True)
        _capture(app, bmp_png, "04-bmp-to-png")

        scanning = _layer_dialog()
        scanning.name_edit.setText("Metal Scan")
        scanning.disk_source.edit.setText(str(jpg))
        scanning.scan_button.setEnabled(False)
        scanning.scan_state.setText("Сканирование… найдено файлов: 18 420")
        _capture(app, scanning, "05-scanning")

        error = _layer_dialog()
        error.name_edit.setText("Metal Mixed")
        error.disk_source.edit.setText(str(mixed))
        error._scan_succeeded(scan_layer_source(mixed, maximum_frames=16))
        _capture(app, error, "06-blocking-error")

        progress = QProgressDialog(
            "Скопировано и обработано: 12 460 из 31 200\nframe_12460.bmp",
            "Отменить",
            0,
            31_200,
        )
        progress.setWindowTitle("Импорт слоя")
        progress.setValue(12_460)
        progress.setMinimumDuration(0)
        progress.resize(620, 150)
        _capture(app, progress, "07-import-progress")

        success = QMessageBox(
            QMessageBox.Icon.Information,
            "Импорт завершён",
            "Слой «Metal 1» создан атомарно.\nИсходная папка не изменена.",
            QMessageBox.StandardButton.Ok,
        )
        _capture(app, success, "08-import-success")

        unavailable = QMessageBox(
            QMessageBox.Icon.Warning,
            "Хранилище недоступно",
            "Метаданные и история доступны только для просмотра. "
            "Файловые операции и плагины заблокированы.\n\n"
            r"Z:\KrakenSource\Chip A",
            QMessageBox.StandardButton.Ok,
        )
        _capture(app, unavailable, "09-unavailable-drive")

        layer_manager = LayerManagerDialog("project-1")
        layer_manager.set_layers(
            [LayerListItem("layer-1", "Metal 1", "metal", "#60A5FA")],
            "layer-1",
        )
        image_path = r"D:\Kraken source\Chip A\img\Metal 1"
        layer_manager.set_pipeline(
            LayerPipelineSnapshot(
                "project-1",
                "layer-1",
                (
                    PipelineLane(
                        "source-1",
                        "Исходные изображения",
                        (
                            PipelineNode(
                                "source-1",
                                "Исходные изображения",
                                "source",
                                "Исходные изображения",
                                "source-1",
                                True,
                                details={"источник": image_path},
                            ),
                        ),
                        (),
                    ),
                ),
            )
        )
        _capture(app, layer_manager, "12-layer-image-source")

        returned = QMessageBox(
            QMessageBox.Icon.Information,
            "Результат опубликован",
            "Запуск a19f27c3 сохранён во втором хранилище.\n"
            "Создано новое активное векторное представление.",
            QMessageBox.StandardButton.Ok,
        )
        _capture(app, returned, "10-plugin-return")

        deletion = QInputDialog()
        deletion.setWindowTitle("Удалить слой")
        deletion.setLabelText(
            "Управляемые файлы будут перемещены в корзину Kraken, "
            "внешние папки останутся без изменений.\n\n"
            "Для подтверждения введите: Metal 1"
        )
        deletion.setInputMode(QInputDialog.InputMode.TextInput)
        deletion.setTextEchoMode(QLineEdit.EchoMode.Normal)
        deletion.setOkButtonText("Удалить")
        deletion.setCancelButtonText("Отмена")
        deletion.resize(620, 190)
        _capture(app, deletion, "11-delete-confirmation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
