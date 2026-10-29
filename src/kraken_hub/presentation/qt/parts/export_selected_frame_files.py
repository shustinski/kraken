"""manager_app: export selected frame files."""
from __future__ import annotations
from pathlib import Path
from uuid import uuid4


class ExportSelectedFrameFilesMixin:
    def _export_selected_frame_files(self, context) -> None:
        from datetime import UTC, datetime

        from PyQt6.QtWidgets import QFileDialog, QMessageBox

        parent = QFileDialog.getExistingDirectory(
            self.shell,
            "Каталог для выгрузки файлов",
        )
        if not parent:
            return
        coordinates = tuple(context.selection.coordinates(maximum=10_000))
        destination = Path(parent) / (
            "Кадры_"
            + datetime.now(UTC).astimezone().strftime("%Y%m%d-%H%M%S-")
            + uuid4().hex[:6]
        )
        exported = 0
        try:
            destination.mkdir(parents=False, exist_ok=False)
            for x, y in coordinates:
                files = self._active_frame_files(x, y)
                if not files:
                    continue
                frame_directory = destination / f"{x}_{y}"
                frame_directory.mkdir(exist_ok=False)
                used_names: set[str] = set()
                for label, _series, version in files:
                    filename = version.filename
                    if filename.casefold() in used_names:
                        prefix = "".join(
                            character
                            if character.isalnum() or character in "-_"
                            else "_"
                            for character in label
                        ).strip("_") or "file"
                        filename = f"{prefix}_{filename}"
                    used_names.add(filename.casefold())
                    self.service.export_artifact_version(
                        self._project_id,
                        version,
                        frame_directory / filename,
                    )
                    exported += 1
            if exported == 0:
                destination.rmdir()
                QMessageBox.information(
                    self.shell,
                    "Нет файлов",
                    "У выбранных кадров нет активных файлов для выгрузки.",
                )
                return
        except (OSError, RuntimeError, ValueError) as exc:
            QMessageBox.warning(
                self.shell,
                "Не удалось выгрузить файлы",
                str(exc),
            )
            return
        QMessageBox.information(
            self.shell,
            "Файлы выгружены",
            f"Файлов: {exported}\nКаталог: {destination}",
        )
