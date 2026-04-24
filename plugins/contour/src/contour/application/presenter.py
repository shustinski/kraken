from __future__ import annotations

import json
from pathlib import Path

from ..i18n import active_language
from .model import ContourApplicationModel
from .view import ContourMainView


class ContourPresenter:
    def __init__(self, *, model: ContourApplicationModel, view: ContourMainView) -> None:
        self.model = model
        self.view = view
        self._bind_view()

    def initialize(self) -> None:
        self.view.set_window_title(self.model.window_title)
        self.view.resize_window(*self.model.initial_size)
        self.view.set_ui_language(active_language(self.model.language))
        self._apply_startup_configuration()

    def _bind_view(self) -> None:
        self.view.bind_log_message(self._on_log_message)
        self.view.bind_image_processed(self._on_image_processed)

    def _apply_startup_configuration(self) -> None:
        startup = self.model.startup
        if startup.output_dir:
            self.view.set_output_directory(startup.output_dir)
        if startup.cif_dir:
            self.view.set_cif_directory(startup.cif_dir)
        if startup.pipeline_json:
            self.view.set_pipeline(self._load_pipeline_payload(startup.pipeline_json))

        input_directory = startup.fallback_input_directory
        if input_directory:
            self.view.set_input_directory(input_directory)
        if startup.file_paths:
            self.view.load_images(startup.file_paths)

    def _on_log_message(self, message: str) -> None:
        self.view.show_status_message(message)

    def _on_image_processed(self, image_path: str, polygons: list) -> None:
        self.view.show_status_message(f"{Path(image_path).name}: {len(polygons)} polygons", 5000)

    @staticmethod
    def _load_pipeline_payload(pipeline_json: str) -> dict:
        pipeline_path = Path(pipeline_json)
        return json.loads(pipeline_path.read_text(encoding="utf-8"))
