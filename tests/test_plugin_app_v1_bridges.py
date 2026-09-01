from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from contour.kraken_bridge import (
    ContourKrakenSession,
    KrakenBridgeError as ContourBridgeError,
    prepare_contour_launch,
)
from kraken_core.plugin_protocol import (
    PluginFrameInput,
    PluginJobManifest,
    PluginResultManifest,
)
from neuralimage.kraken_bridge import (
    HeadlessOptions,
    KrakenBridgeError as NeuralBridgeError,
    NeuralImageKrakenSession,
    load_session_from_values,
)


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _manifest(
    *,
    operation: str,
    payload: bytes,
    parameters: dict | None = None,
) -> PluginJobManifest:
    return PluginJobManifest(
        job_id="job-bridge-1",
        operation=operation,
        project_id="project-1",
        layer_id="layer-1",
        actor_id="actor-1",
        target_representation_id="representation-1",
        inputs=(
            PluginFrameInput(
                frame_id="frame-1",
                x=2,
                y=3,
                artifact_version_id="artifact-version-1",
                sha256=_digest(payload),
                media_type="image/png",
                relative_path="inputs/staged-frame.png",
            ),
        ),
        parameters=parameters or {},
    )


def _workspace(root: Path, manifest: PluginJobManifest, payload: bytes) -> tuple[Path, Path]:
    (root / "inputs").mkdir(parents=True)
    (root / "outputs").mkdir()
    (root / "inputs" / "staged-frame.png").write_bytes(payload)
    job_path = root / "job.json"
    job_path.write_text(manifest.to_json(), encoding="utf-8")
    return job_path, root / "result.json"


class ContourBridgeV1Tests(unittest.TestCase):
    def test_environment_launch_uses_exact_manifest_inputs_and_writes_hashed_cif(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = b"image-payload"
            manifest = _manifest(operation="frames.vectorize.v1", payload=payload)
            job_path, result_path = _workspace(root, manifest, payload)
            session, argv = prepare_contour_launch(
                ["--language", "en"],
                environ={
                    "KRAKEN_JOB_MANIFEST": str(job_path),
                    "KRAKEN_RESULT_MANIFEST": str(result_path),
                    "KRAKEN_STAGING_ROOT": str(root),
                },
            )
            self.assertIsNotNone(session)
            assert session is not None
            self.assertEqual([root / "inputs" / "staged-frame.png"], list(session.input_paths))
            self.assertEqual(str(root / "inputs" / "staged-frame.png"), argv[-1])
            self.assertEqual(str(root / "outputs"), argv[-2])

            cif_payload = b"DS 1 1 1;\nDF;\nE\n"
            (root / "outputs" / "staged-frame.cif").write_bytes(cif_payload)
            written = session.write_result()
            restored = PluginResultManifest.from_json(result_path.read_text(encoding="utf-8"))
            self.assertEqual("succeeded", written.outcome)
            self.assertEqual(_digest(cif_payload), restored.outputs[0].sha256)
            self.assertEqual("outputs/staged-frame.cif", restored.outputs[0].relative_path)

    def test_dataset_preparation_fills_dataset_path_instead_of_result_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = b"image-payload"
            manifest = _manifest(operation="frames.dataset.prepare.v1", payload=payload)
            job_path, result_path = _workspace(root, manifest, payload)

            session, argv = prepare_contour_launch(
                [],
                environ={
                    "KRAKEN_JOB_MANIFEST": str(job_path),
                    "KRAKEN_RESULT_MANIFEST": str(result_path),
                    "KRAKEN_STAGING_ROOT": str(root),
                },
            )

            self.assertIsNotNone(session)
            self.assertIn("--dataset-dir", argv)
            self.assertNotIn("--output-dir", argv)
            self.assertEqual(str(root / "outputs"), argv[argv.index("--dataset-dir") + 1])
            self.assertEqual(str(root / "inputs" / "staged-frame.png"), argv[-1])

    def test_managed_mode_rejects_arbitrary_cli_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = b"image"
            job_path, result_path = _workspace(
                root,
                _manifest(operation="frames.vectorize.v1", payload=payload),
                payload,
            )
            with self.assertRaises(ContourBridgeError):
                prepare_contour_launch(
                    ["outside-project.png"],
                    environ={
                        "KRAKEN_JOB_MANIFEST": str(job_path),
                        "KRAKEN_RESULT_MANIFEST": str(result_path),
                        "KRAKEN_STAGING_ROOT": str(root),
                    },
                )

    def test_checksum_and_protocol_mismatch_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = b"image"
            manifest = _manifest(operation="frames.vectorize.v1", payload=payload)
            job_path, result_path = _workspace(root, manifest, b"tampered")
            with self.assertRaises(ContourBridgeError):
                ContourKrakenSession.load(
                    job_manifest=job_path,
                    result_manifest=result_path,
                    staging_root=root,
                )

            (root / "inputs" / "staged-frame.png").write_bytes(payload)
            raw = manifest.to_dict()
            raw["protocol_version"] = "2.0"
            job_path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(ContourBridgeError):
                ContourKrakenSession.load(
                    job_manifest=job_path,
                    result_manifest=result_path,
                    staging_root=root,
                )

            raw["protocol_version"] = "1.0"
            raw["inputs"][0]["relative_path"] = "../outside.png"
            job_path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(ContourBridgeError):
                ContourKrakenSession.load(
                    job_manifest=job_path,
                    result_manifest=result_path,
                    staging_root=root,
                )


class NeuralImageBridgeV1Tests(unittest.TestCase):
    def test_protocol_mismatch_is_rejected_before_inference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = b"image"
            manifest = _manifest(operation="frames.binary-segment.v1", payload=payload)
            job_path, result_path = _workspace(root, manifest, payload)
            raw = manifest.to_dict()
            raw["protocol_version"] = "99.0"
            job_path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(NeuralBridgeError):
                NeuralImageKrakenSession.load(
                    job_manifest=job_path,
                    result_manifest=result_path,
                    staging_root=root,
                )

    def test_missing_model_returns_failed_manifest_without_fake_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = b"image"
            job_path, result_path = _workspace(
                root,
                _manifest(operation="frames.binary-segment.v1", payload=payload),
                payload,
            )
            session = NeuralImageKrakenSession.load(
                job_manifest=job_path,
                result_manifest=result_path,
                staging_root=root,
            )
            result = session.run_headless()
            self.assertEqual("failed", result.outcome)
            self.assertEqual((), result.outputs)
            self.assertIn("model_relative_path", result.errors[0])
            self.assertTrue(result_path.is_file())

    def test_model_and_inputs_are_hash_pinned_inside_staging(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = b"image"
            model_payload = b"model"
            parameters = {
                "model_relative_path": "models/model.ckpt",
                "model_sha256": _digest(model_payload),
                "model_version": "test-model-1",
                "patch_size": [128, 256],
                "batch_size": 2,
                "overlap": 16,
                "threshold": 0.6,
                "use_auto_threshold": False,
                "tta": True,
            }
            job_path, result_path = _workspace(
                root,
                _manifest(
                    operation="frames.binary-segment.v1",
                    payload=payload,
                    parameters=parameters,
                ),
                payload,
            )
            (root / "models").mkdir()
            (root / "models" / "model.ckpt").write_bytes(model_payload)
            session = NeuralImageKrakenSession.load(
                job_manifest=job_path,
                result_manifest=result_path,
                staging_root=root,
            )
            options = HeadlessOptions.from_session(session)
            self.assertEqual((128, 256), options.patch_size)
            self.assertEqual("test-model-1", options.model_version)
            self.assertTrue(options.tta)

            parameters["model_sha256"] = "0" * 64
            result_path.unlink(missing_ok=True)
            job_path.write_text(
                _manifest(
                    operation="frames.binary-segment.v1",
                    payload=payload,
                    parameters=parameters,
                ).to_json(),
                encoding="utf-8",
            )
            bad_session = NeuralImageKrakenSession.load(
                job_manifest=job_path,
                result_manifest=result_path,
                staging_root=root,
            )
            with self.assertRaises(NeuralBridgeError):
                HeadlessOptions.from_session(bad_session)

    def test_cli_environment_is_optional_for_standalone_and_rejects_partial_values(self) -> None:
        self.assertIsNone(
            load_session_from_values(
                job_manifest=None,
                result_manifest=None,
                staging_root=None,
                environ={},
            )
        )
        with self.assertRaises(NeuralBridgeError):
            load_session_from_values(
                job_manifest="job.json",
                result_manifest=None,
                staging_root=None,
                environ={},
            )


if __name__ == "__main__":
    unittest.main()
