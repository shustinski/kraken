"""processing_mixin: export dataset frame state."""
from __future__ import annotations
from typing import Any
from contour.widget_parts._imports import (
    ImageProcessingState,
    PolygonData,
    PreprocessingPipeline,
    export_frame_to_dataset,
)


class ExportDatasetFrameStateMixin:
    def set_pipeline(self: Any, config: dict) -> None:
        self._pipeline = PreprocessingPipeline.from_dict(config)
        self._populate_pipeline_list()
        self._auto_apply_pipeline()

    def get_pipeline(self: Any) -> dict:
        return self._pipeline.to_dict()

    def process_current_image(self: Any, *_args, debounced: bool = False) -> None:
        self._queue_preview_processing(debounced=debounced)

    def _process_current_image_and_save(self: Any, *_args) -> None:
        """Process the selected frame and persist the enabled result formats."""

        self._queue_preview_processing(debounced=False, save_result=True)

    def _export_dataset_frame_for_state(
        self: Any,
        image_path: str,
        state: ImageProcessingState,
        polygons: list[PolygonData],
        dataset_directory: str | None = None,
    ) -> dict[str, str]:
        target_directory = dataset_directory or self.dataset_dir_edit.text().strip()
        result = export_frame_to_dataset(
            dataset_directory=target_directory,
            image_path=image_path,
            state=state,
            polygons=polygons,
        )
        if result.message_key is not None:
            message = self._tr(result.message_key, **(result.message_kwargs or {}))
            self._append_log(message)
            if not result.saved_files:
                self._last_vector_persist_error = message
        return result.saved_files

    def export_current_frame_to_dataset(self: Any, dataset_directory: str | None = None) -> dict[str, str]:
        current_state = self._workspace.current_state
        current_image_path = self._workspace.current_image_path
        if current_state is None or current_image_path is None:
            self._append_log(self._tr("nothing_to_save_log"))
            return {}
        current_polygons = self.get_polygons()
        self._workspace.update_current_polygons(current_polygons)
        self._update_frame_item_status(current_image_path)
        return self._export_dataset_frame_for_state(
            current_image_path,
            current_state,
            current_polygons,
            dataset_directory=dataset_directory,
        )
