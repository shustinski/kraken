from __future__ import annotations

import gzip
import hashlib
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from kraken_agent.runner import ANALYSIS_OPERATION, PluginProcessSpec, PluginRegistry, SubprocessPluginRunner
from kraken_agent.service import AgentControlServer
from kraken_core.analysis_protocol import (
    AnalysisArtifactInput,
    AnalysisFrameInput,
    AnalysisFrameResult,
    AnalysisMetricValue,
    AnalysisOutcome,
    AnalysisSourceRole,
)
from kraken_core.analysis_run_protocol import (
    AnalysisExpression,
    AnalysisPartitionJobManifest,
    AnalysisPartitionResultManifest,
    AnalysisRecipe,
    AnalysisRecordBundle,
    AnalysisRuntimeIdentity,
    AnalysisSourceBinding,
    canonical_json,
)
from kraken_manager.application.analysis_runs import AnalysisGatewayJob, AnalysisRunCoordinator
from kraken_manager.infrastructure.analysis import FilesystemAnalysisStore
from kraken_manager.infrastructure.plugin import AgentAnalysisGateway


class FakeAnalysisGateway:
    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root
        self.jobs: dict[str, AnalysisGatewayJob] = {}
        self.manifests: dict[str, AnalysisPartitionJobManifest] = {}
        self.results: dict[str, tuple[AnalysisPartitionResultManifest, Path]] = {}
        self.reject_submissions = False

    def submit(self, manifest, source_paths):
        if self.reject_submissions:
            raise RuntimeError("agent unavailable")
        assert {artifact.artifact_version_id for frame in manifest.frames for artifact in frame.artifacts} <= set(
            source_paths
        )
        job = AnalysisGatewayJob(manifest.job_id, "importing", 2)
        self.jobs[manifest.job_id] = job
        self.manifests[manifest.job_id] = manifest
        frames = tuple(
            AnalysisFrameResult(
                frame.frame_id,
                frame.x,
                frame.y,
                "ready",
                (AnalysisMetricValue("iou", 1.0, 1.0),),
            )
            for frame in manifest.frames
        )
        bundle_path = self.output_root / f"{manifest.job_id}.jsonl.gz"
        uncompressed_size = 0
        with bundle_path.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as archive:
                for frame in frames:
                    encoded = (canonical_json(frame.to_payload()) + "\n").encode()
                    archive.write(encoded)
                    uncompressed_size += len(encoded)
        bundle = AnalysisRecordBundle(
            f"outputs/{bundle_path.name}",
            hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
            bundle_path.stat().st_size,
            uncompressed_size,
            len(frames),
        )
        result = AnalysisPartitionResultManifest(
            manifest.job_id,
            manifest.run_id,
            manifest.partition_id,
            manifest.project_id,
            AnalysisOutcome.SUCCEEDED,
            bundle,
            worker_runtime=AnalysisRuntimeIdentity("workflow-test-fixture", "sha256:" + "0" * 64,
                "fixture-python", "fixture-numpy", "fixture-opencv", "fixture-os"),
        )
        self.results[manifest.job_id] = (result, bundle_path)
        return job

    def get(self, job_id):
        return self.jobs[job_id]

    def result(self, manifest):
        return self.results[manifest.job_id]

    def confirm_partial(self, job):
        return AnalysisGatewayJob(job.job_id, "importing", job.revision + 1)

    def complete_import(self, job):
        completed = AnalysisGatewayJob(job.job_id, "succeeded", job.revision + 1)
        self.jobs[job.job_id] = completed
        return completed

    def cancel(self, job):
        cancelled = AnalysisGatewayJob(job.job_id, "cancelled", job.revision + 1)
        self.jobs[job.job_id] = cancelled
        return cancelled


def _input_frames(source: Path, count: int) -> tuple[AnalysisFrameInput, ...]:
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return tuple(
        AnalysisFrameInput(
            f"frame-{index:04d}",
            index + 1,
            1,
            tuple(
                AnalysisArtifactInput(
                    key,
                    AnalysisSourceRole.MODEL_OUTPUT,
                    f"series-{key}",
                    f"version-{key}",
                    f"inputs/{key}/frame-{index:04d}.png",
                    "image/png",
                    digest,
                )
                for key in ("A", "B")
            ),
        )
        for index in range(count)
    )


def _coordinator(tmp_path: Path, *, count: int = 1001):
    source = tmp_path / "source.png"
    source.write_bytes(b"immutable")
    store = FilesystemAnalysisStore(tmp_path / "catalog", "project-1")
    gateway = FakeAnalysisGateway(tmp_path)
    coordinator = AnalysisRunCoordinator(store, gateway, lambda _version: source)
    recipe = AnalysisRecipe(
        AnalysisExpression.binary("compare", AnalysisExpression.source("A"), AnalysisExpression.source("B"))
    )
    run = coordinator.start(
        project_id="project-1",
        frames=_input_frames(source, count),
        source_bindings=(
            AnalysisSourceBinding("A", "model-a", "v1"),
            AnalysisSourceBinding("B", "model-b", "v1"),
        ),
        recipe=recipe,
        runtime=AnalysisRuntimeIdentity("1", "build", "3.14", "2", "5", "windows"),
    )
    return coordinator, store, gateway, run


def test_coordinator_dispatches_and_progressively_imports_all_partitions(tmp_path) -> None:
    coordinator, store, gateway, run = _coordinator(tmp_path)

    assert len(gateway.jobs) == 2
    assert store.get_run(run.run_id).state == "running"
    refreshed = coordinator.refresh(run.run_id)

    assert refreshed.state == "completed"
    assert refreshed.completed_frames == 1001
    assert len(store.frame_results(run.run_id, "iou")) == 1001
    assert {job.state for job in gateway.jobs.values()} == {"succeeded"}


def test_repeat_has_new_run_id_but_same_fingerprint(tmp_path) -> None:
    coordinator, store, _gateway, run = _coordinator(tmp_path, count=2)
    coordinator.refresh(run.run_id)

    repeated = coordinator.repeat(run.run_id)

    assert repeated.run_id != run.run_id
    assert repeated.fingerprint == run.fingerprint
    assert len(store.list_runs()) == 2


def test_failed_submission_is_reissued_with_a_new_agent_job_id(tmp_path) -> None:
    coordinator, store, gateway, completed_run = _coordinator(tmp_path, count=1)
    coordinator.refresh(completed_run.run_id)
    source = tmp_path / "source.png"
    frames = _input_frames(source, 1)
    recipe = completed_run.recipe
    gateway.reject_submissions = True
    with pytest.raises(RuntimeError, match="agent unavailable"):
        coordinator.start(
            project_id="project-1",
            frames=frames,
            source_bindings=completed_run.source_bindings,
            recipe=recipe,
            runtime=completed_run.runtime,
        )
    failed = next(item for item in store.list_runs() if item.state == "failed")
    old_job_id = store.failed_partitions(failed.run_id)[0].job_id
    gateway.reject_submissions = False

    coordinator.retry_failed(failed.run_id)
    replacement = store.retryable_partitions(failed.run_id)[0]

    assert replacement.job_id != old_job_id
    assert replacement.job_id in gateway.jobs


def test_local_agent_worker_gateway_and_manager_import_smoke(tmp_path, monkeypatch, qtbot) -> None:
    source = tmp_path / "mask.png"
    assert cv2.imwrite(str(source), np.asarray([[0, 255]], dtype=np.uint8))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    artifacts = tuple(
        AnalysisArtifactInput(
            key,
            AnalysisSourceRole.MODEL_OUTPUT,
            f"series-{key}",
            f"version-{key}",
            f"inputs/{key}/frame.png",
            "image/png",
            digest,
        )
        for key in ("A", "B")
    )
    control = AgentControlServer.create(tmp_path / "agent.sqlite3", token="test-token")
    httpd = control.build_http_server()
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()
    try:
        gateway = AgentAnalysisGateway(
            base_url=f"http://127.0.0.1:{control.port}",
            token="test-token",
            staging_root=tmp_path / "staging",
        )
        store = FilesystemAnalysisStore(tmp_path / "catalog", "project-1")
        coordinator = AnalysisRunCoordinator(store, gateway, lambda _version: source)
        run = coordinator.start(
            project_id="project-1",
            frames=(AnalysisFrameInput("frame-1", 1, 1, artifacts),),
            source_bindings=(
                AnalysisSourceBinding("A", "model-a", "v1"),
                AnalysisSourceBinding("B", "model-b", "v1"),
            ),
            recipe=AnalysisRecipe(
                AnalysisExpression.binary(
                    "compare", AnalysisExpression.source("A"), AnalysisExpression.source("B")
                )
            ),
            runtime=AnalysisRuntimeIdentity("1", "build", "3.14", "2", "5", "windows"),
        )
        runner = SubprocessPluginRunner(
            control.store,
            tmp_path / "staging",
            PluginRegistry(
                {
                    ANALYSIS_OPERATION: PluginProcessSpec(
                        ANALYSIS_OPERATION,
                        (sys.executable, "-m", "karakal.worker"),
                    )
                }
            ),
        )

        assert runner.run_once() is True
        completed = coordinator.refresh(run.run_id)

        assert completed.state == "completed"
        assert completed.completed_frames == 1
        from karakal.runtime_identity import worker_runtime_identity
        assert completed.worker_runtime == worker_runtime_identity()
        transported = gateway._request("GET", f"/api/v1/jobs/{store.all_partitions(run.run_id)[0].job_id}/result")
        assert transported["worker_runtime"] == completed.worker_runtime.to_payload()
        assert len(store.frame_results(run.run_id, "iou")) == 1
        before_metrics = tuple(dict(row) for row in store.frame_results(run.run_id, "iou"))
        from kraken_manager.infrastructure.plugin.agent_gateway import AgentUnavailable

        with monkeypatch.context() as scoped:
            scoped.setattr(gateway, "submit", lambda *_args: (_ for _ in ()).throw(AgentUnavailable("injected map submit outage")))
            task = coordinator.request_map(run.run_id, "frame-1", idempotency_key="local-map")
        assert task["state"] == "queued"
        map_job_id = task["manifest"]["job_id"]
        store.rebuild()
        reopened = FilesystemAnalysisStore(tmp_path / "catalog", "project-1")
        recovered = AnalysisRunCoordinator(reopened, gateway, lambda _version: source)
        recovered.refresh_maps(run.run_id)
        assert reopened.map_tasks(run.run_id)["frame-1"]["manifest"]["job_id"] == map_job_id
        assert runner.run_once(job_id=map_job_id)
        with monkeypatch.context() as scoped:
            scoped.setattr(gateway, "complete_import", lambda *_args: (_ for _ in ()).throw(AgentUnavailable("injected acknowledgement outage")))
            recovered.refresh_maps(run.run_id)
        task = reopened.map_tasks(run.run_id)["frame-1"]
        assert task["state"] == "ready" and not task["acknowledged"]
        data = reopened.map_bytes(run.run_id, "frame-1")
        assert data.startswith(b"\x89PNG")
        reopened.database.unlink()
        rebuilt = FilesystemAnalysisStore(tmp_path / "catalog", "project-1")
        restarted = AnalysisRunCoordinator(rebuilt, gateway, lambda _version: source)
        restarted.refresh_maps(run.run_id)
        assert rebuilt.map_tasks(run.run_id)["frame-1"]["acknowledged"]
        assert gateway.get(map_job_id).state == "succeeded"
        map_path = tmp_path / "staging" / map_job_id / task["result"]["rendered_map"]["relative_path"]
        assert map_path.resolve().is_relative_to(tmp_path.resolve())
        map_path.unlink()
        assert rebuilt.map_bytes(run.run_id, "frame-1") == data
        assert tuple(dict(row) for row in rebuilt.frame_results(run.run_id, "iou")) == before_metrics
        assert restarted.request_map(run.run_id, "frame-1", idempotency_key="show-cached-map")["manifest"]["job_id"] == map_job_id
        from kraken_manager.infrastructure.migration import CanonicalBundleExporter, CanonicalBundleImporter

        backup = tmp_path / "analysis-with-map-backup"
        CanonicalBundleExporter(rebuilt.events, rebuilt.blobs).export(backup)
        restored = FilesystemAnalysisStore(tmp_path / "restored-catalog", "project-1")
        CanonicalBundleImporter(restored.events, restored.blobs).import_bundle(backup)
        restored.rebuild()
        assert restored.map_bytes(run.run_id, "frame-1") == data
        assert tuple(dict(row) for row in restored.frame_results(run.run_id, "iou")) == before_metrics
        from PyQt6.QtWidgets import QWidget, QVBoxLayout, QDialog, QLabel, QPlainTextEdit
        from PyQt6.QtCore import pyqtSignal
        from types import SimpleNamespace
        from kraken_hub.manager_app import DesktopController
        from kraken_manager.presentation.qt.analysis_ui import AnalysisRunsPanel

        class Workspace(QWidget):
            analysisRequested = pyqtSignal(object)

        widget = Workspace()
        panel = widget.analysis_runs_panel = AnalysisRunsPanel(widget)
        QVBoxLayout(widget).addWidget(panel)
        qtbot.addWidget(widget)
        widget.show()
        desktop = object.__new__(DesktopController)
        desktop.language = "English"
        desktop._background_workers = set()
        desktop.service = SimpleNamespace(catalog_root=rebuilt.catalog_root, data_dir=tmp_path,
            artifact_source_path=lambda *_args: source, is_remote_project=lambda _id: False)
        desktop.agent_runtime = SimpleNamespace(ensure_started=lambda: (), base_url=gateway.base_url, token=gateway.token)
        desktop._configure_analysis(widget, "project-1")
        maps = widget._analysis_map_controller
        qtbot.waitUntil(lambda: maps._worker is None)
        panel.result_table.selectRow(0)
        panel.map_button.click()
        qtbot.waitUntil(lambda: widget.findChild(QDialog, "localAnalysisMapDialog") is not None, timeout=5000)
        dialog = widget.findChild(QDialog, "localAnalysisMapDialog")
        image = dialog.findChild(QLabel, "analysisMapImage").pixmap()
        assert dialog.isVisible() and not dialog.isModal()
        assert image.width() == 2 and image.toImage().pixelColor(1, 0).red() == 255
        assert "version-A" in dialog.findChild(QPlainTextEdit, "analysisMapInputs").toPlainText()
        assert completed.worker_runtime.engine_build in dialog.findChild(QPlainTextEdit, "analysisMapRuntime").toPlainText()
        qtbot.waitUntil(lambda: maps._worker is None)
        assert not desktop._background_workers
        maps.timer.stop()
        dialog.close()
        widget.close()
        from dataclasses import replace
        incompatible = replace(completed.worker_runtime, numpy_version="incompatible-fixture")
        refused = coordinator.start(project_id=run.project_id, frames=(AnalysisFrameInput("frame-1", 1, 1, artifacts),),
            source_bindings=run.source_bindings, recipe=run.recipe, runtime=run.runtime, expected_runtime=incompatible)
        assert runner.run_once()
        failed = coordinator.refresh(refused.run_id)
        assert failed.state == "failed" and failed.worker_runtime is None
        assert not store.frame_results(refused.run_id, "iou")
        history = store.events.load_stream(f"analysis-run:{refused.run_id}")
        failure_event = next(event for event in history if event.data.get("payload", {}).get("worker_result"))
        assert failure_event.data["payload"]["worker_result"]["worker_runtime"] == completed.worker_runtime.to_payload()
        assert "pinned requested runtime" in failure_event.data["payload"]["error"]
        failed_transport = gateway._request("GET", f"/api/v1/jobs/{store.all_partitions(refused.run_id)[0].job_id}/result")
        assert failed_transport["outcome"] == "failed"
        assert failed_transport["worker_runtime"] == completed.worker_runtime.to_payload()
        repeated = coordinator.repeat(refused.run_id)
        assert store.all_partitions(repeated.run_id)[0].expected_runtime == incompatible
        desktop._refresh_analysis_panel(widget)
        from PyQt6.QtCore import Qt
        for index in range(panel.run_table.rowCount()):
            if panel.run_table.item(index, 0).data(Qt.ItemDataRole.UserRole) == refused.run_id:
                panel.run_table.selectRow(index)
                break
        widget.show()
        panel.runtime_button.click()
        qtbot.waitUntil(lambda: widget.findChild(QDialog, "analysisRuntimeDialog") is not None, timeout=5000)
        runtime_dialog = widget.findChild(QDialog, "analysisRuntimeDialog")
        report_text = runtime_dialog.findChild(QPlainTextEdit, "analysisRuntimeReport").toPlainText()
        assert "pinned requested runtime" in report_text and "incompatible-fixture" in report_text
        assert completed.worker_runtime.engine_build in report_text
        qtbot.waitUntil(lambda: maps._worker is None)
        runtime_dialog.close()
        widget.close()
    finally:
        httpd.shutdown()
        httpd.server_close()
        server_thread.join(timeout=5)
