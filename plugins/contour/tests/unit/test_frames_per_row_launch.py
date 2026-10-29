"""Project «Кадров в строке» is forwarded into the frame matrix on open."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from contour.application.frame_lod import FixedGridFrameLayout
from contour.application.frame_matrix_layout import (
    frame_matrix_cell,
    matrix_layout_columns,
    with_frames_per_row_argument,
)
from contour.application.model import StartupConfiguration
from contour.kraken_bridge import prepare_contour_launch
from kraken_core.plugin_protocol import PluginFrameInput, PluginJobManifest


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class FramesPerRowLaunchTests(unittest.TestCase):
    def test_project_width_is_the_matrix_column_count(self) -> None:
        project_width = 12
        startup = StartupConfiguration.from_cli(frames_per_row=project_width)
        self.assertEqual(12, startup.frames_per_row)
        self.assertIsNone(StartupConfiguration.from_cli().frames_per_row)
        self.assertIsNone(StartupConfiguration.from_cli(frames_per_row=0).frames_per_row)

        columns = matrix_layout_columns(startup.frames_per_row or 0)
        self.assertEqual((1, 1), frame_matrix_cell(13, columns))
        layout = FixedGridFrameLayout(frame_count=30, columns=columns, frame_store=None)
        self.assertEqual((1, 1), layout.frame_id_to_row_col(13))

    def test_open_command_forwards_project_width_and_does_not_invent_one(self) -> None:
        self.assertEqual(
            ["--frames-per-row", "12", "--language", "en"],
            with_frames_per_row_argument(["--language", "en"], 12),
        )
        self.assertEqual(
            ["--language", "en"],
            with_frames_per_row_argument(["--language", "en"], None),
        )

        payload = b"image-payload"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "inputs").mkdir()
            (root / "outputs").mkdir()
            (root / "inputs" / "staged-frame.png").write_bytes(payload)
            manifest = PluginJobManifest(
                job_id="job-frames-1",
                operation="frames.vectorize.v1",
                project_id="project-1",
                layer_id="layer-1",
                actor_id="actor-1",
                target_representation_id="representation-1",
                inputs=(
                    PluginFrameInput(
                        frame_id="frame-1",
                        x=1,
                        y=1,
                        artifact_version_id="artifact-version-1",
                        sha256=_digest(payload),
                        media_type="image/png",
                        relative_path="inputs/staged-frame.png",
                    ),
                ),
                parameters={"project_width": 12},
            )
            job_path = root / "job.json"
            job_path.write_text(manifest.to_json(), encoding="utf-8")
            _session, argv = prepare_contour_launch(
                ["--language", "en"],
                environ={
                    "KRAKEN_JOB_MANIFEST": str(job_path),
                    "KRAKEN_RESULT_MANIFEST": str(root / "result.json"),
                    "KRAKEN_STAGING_ROOT": str(root),
                },
            )
            self.assertEqual(["--frames-per-row", "12"], argv[:2])
            self.assertEqual(str(root / "inputs" / "staged-frame.png"), argv[-1])

            job_path.write_text(replace(manifest, parameters={}).to_json(), encoding="utf-8")
            _session, argv = prepare_contour_launch(
                ["--language", "en"],
                environ={
                    "KRAKEN_JOB_MANIFEST": str(job_path),
                    "KRAKEN_RESULT_MANIFEST": str(root / "result.json"),
                    "KRAKEN_STAGING_ROOT": str(root),
                },
            )
            self.assertNotIn("--frames-per-row", argv)


if __name__ == "__main__":
    unittest.main()
