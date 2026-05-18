"""Unit tests for the application service layer (pipeline / dataset / preview / batch)."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from contour.application.services import (
    BatchController,
    BatchStartRequest,
    PathSettingsController,
    PreviewOrchestrator,
    export_frame_to_dataset,
    load_pipeline_config_from_path,
    save_pipeline_config_to_path,
)
from contour.application.dto import PersistedPaths


class PipelineControllerTests(unittest.TestCase):
    def test_roundtrip_preserves_structure(self) -> None:
        config = {"steps": [{"name": "threshold", "parameters": {"value": 128}}]}
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "pipeline.json"
            save_pipeline_config_to_path(path, config)
            loaded = load_pipeline_config_from_path(path)
        self.assertEqual(loaded, config)


class DatasetExporterTests(unittest.TestCase):
    def test_empty_directory_returns_dedicated_message(self) -> None:
        state = _FakeState(np.zeros((8, 8), dtype=np.uint8))
        result = export_frame_to_dataset(
            dataset_directory="   ",
            image_path="x.png",
            state=state,
            polygons=[],
        )
        self.assertEqual(result.saved_files, {})
        self.assertEqual(result.message_key, "dataset_directory_not_set_log")

    def test_unexpected_error_is_surfaced_via_message_key(self) -> None:
        from contour.application.services import dataset_exporter as module

        def _explode(**_kwargs: Any) -> dict[str, str]:
            raise RuntimeError("disk full")

        original = module.export_dataset_frame
        module.export_dataset_frame = _explode
        try:
            result = module.export_frame_to_dataset(
                dataset_directory="out",
                image_path="frame.png",
                state=_FakeState(np.zeros((8, 8), dtype=np.uint8)),
                polygons=[],
            )
        finally:
            module.export_dataset_frame = original

        self.assertEqual(result.saved_files, {})
        self.assertEqual(result.message_key, "dataset_export_failed_log")


class PathSettingsControllerTests(unittest.TestCase):
    def test_save_and_load_roundtrip_through_store(self) -> None:
        store = _FakePathStore()
        controller = PathSettingsController(store)
        paths = PersistedPaths(input_directory="in", output_directory="out")

        controller.save(paths)

        self.assertEqual(controller.load(), paths)

    def test_validate_input_directory_normalizes_and_reports_availability(self) -> None:
        with TemporaryDirectory() as tmp:
            controller = PathSettingsController(_FakePathStore())

            result = controller.validate_input_directory(Path(tmp))

        self.assertTrue(result.available)
        self.assertTrue(result.is_directory)

    def test_validate_input_directory_rejects_missing_path(self) -> None:
        controller = PathSettingsController(_FakePathStore())

        result = controller.validate_input_directory("definitely-missing-directory")

        self.assertFalse(result.available)


class PreviewOrchestratorTests(unittest.TestCase):
    def test_request_ids_are_monotonic(self) -> None:
        orchestrator = PreviewOrchestrator()
        self.assertEqual(orchestrator.next_prepared_id(), 1)
        self.assertEqual(orchestrator.next_prepared_id(), 2)
        self.assertEqual(orchestrator.next_preview_id(), 1)
        self.assertEqual(orchestrator.current_preview_id, 1)
        self.assertEqual(orchestrator.next_auto_tune_id(), 1)


class BatchControllerTests(unittest.TestCase):
    def test_start_delegates_and_respects_busy_state(self) -> None:
        processor = _FakeProcessor()
        controller = BatchController(processor)

        request = _batch_request(["a.png", "b.png"], output_directory="out", save_cif=True)
        self.assertTrue(controller.start(request))
        self.assertTrue(controller.progress_enabled)
        self.assertEqual(processor.start_calls, 1)

        processor.is_running = True
        self.assertFalse(controller.start(request))
        self.assertEqual(processor.start_calls, 1)

    def test_progress_disabled_without_save_cif(self) -> None:
        processor = _FakeProcessor()
        controller = BatchController(processor)
        self.assertTrue(controller.start(_batch_request(["a.png"], output_directory="out", save_cif=False)))
        self.assertFalse(controller.progress_enabled)

    def test_stop_forwards(self) -> None:
        processor = _FakeProcessor()
        BatchController(processor).stop()
        self.assertEqual(processor.stop_calls, 1)


class _FakeState:
    def __init__(self, source_image: np.ndarray) -> None:
        self.source_image = source_image


class _FakeProcessor:
    def __init__(self) -> None:
        self.is_running = False
        self.start_calls = 0
        self.stop_calls = 0

    def start(self, **_kwargs: Any) -> None:
        self.start_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1


class _FakePathStore:
    def __init__(self) -> None:
        self.paths = PersistedPaths()

    def load(self) -> PersistedPaths:
        return self.paths

    def save(self, paths: PersistedPaths) -> None:
        self.paths = paths


def _batch_request(paths: list[str], *, output_directory: str, save_cif: bool) -> BatchStartRequest:
    return BatchStartRequest(
        image_paths=paths,
        pipeline_config={},
        contour_settings=_FakeContourSettings(),
        display_settings=_FakeDisplaySettings(),
        save_options=_FakeSaveOptions(save_cif=save_cif),
        output_directory=output_directory,
        max_workers=1,
    )


class _FakeContourSettings: ...


class _FakeDisplaySettings: ...


class _FakeSaveOptions:
    def __init__(self, *, save_cif: bool) -> None:
        self.save_cif = save_cif


if __name__ == "__main__":
    unittest.main()
