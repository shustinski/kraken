"""Isolated, reviewable Desktop deployment probe (including its bundled Agent)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
import traceback
from uuid import uuid4


def run_self_test(output_directory, *, reference_manifest=None):
    base = Path(output_directory).expanduser().resolve()
    run = base / f"desktop-self-test-{uuid4().hex}"
    run.mkdir(parents=True)
    report = {"status": "running", "directory": str(run), "checks": []}
    store = None
    controller = None
    dialogs = []
    app = None
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtGui import QFont, QFontDatabase
        from kraken_core.settings import QSettings, configure_settings_directory
        from kraken_hub.infrastructure.credentials.secret_store import KeyringSecretStore
        from kraken_hub.composition import EmbeddedProjectService
        from kraken_hub.presentation.qt.manager_app import DesktopController
        from kraken_hub.presentation.qt.work_planning_dialog import WorkPlanningDialog
        from kraken_hub.presentation.qt.layer_pipeline_dialog import LayerPipelineDialog
        from kraken_hub.presentation.qt.quality_dialog import QualityDialog
        from kraken_manager.presentation.qt import ProjectManagerShell, ProjectListItem
        from kraken_manager.domain.project import GridOrientation
        from kraken_hub.presentation.probes.deployment_matrix_probe import (
            advance_head, prepare_project, probe_records, verify_project, verify_workspace,
        )

        class ProbeStore(KeyringSecretStore):
            def __init__(self):
                super().__init__(f"Kraken.SelfTest.{uuid4().hex}")
                self.created = set()

            def set(self, key, value):
                super().set(key, value)
                self.created.add(key)

        store = ProbeStore()
        secret = os.urandom(32)
        store.set("probe", secret)
        if store.get("probe") != secret:
            raise RuntimeError("OS secret-store round trip failed")
        report["checks"].append("OS secret-store round trip")
        app = QApplication.instance() or QApplication([])
        configure_settings_directory(run / "settings")
        settings_file = Path(QSettings("Kraken", "KrakenHub").fileName()).resolve()
        if not settings_file.is_relative_to((run / "settings").resolve()):
            raise RuntimeError("Deployment settings did not leave the user's global settings namespace")
        report["settings_isolated"] = True
        report["settings_file"] = str(settings_file)
        if not QFontDatabase.families() and os.name == "nt":
            font = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/segoeui.ttf"
            if font.is_file() and QFontDatabase.addApplicationFont(str(font)) >= 0:
                app.setFont(QFont("Segoe UI", 9))
        service = EmbeddedProjectService(run / "profile", secret_store=store)
        session = service.create_initial_account("self-test", "Deployment self-test", "")
        records = probe_records(run, reference_manifest)
        size = 4 if reference_manifest else 2
        project = service.create_project(
            principal=session.principal,
            name="Deployment probe",
            width=size,
            height=size,
            orientation=GridOrientation.Y_DOWN,
            idempotency_key="project",
            layer_template=True,
        )
        report["managed_matrix"] = prepare_project(service, session, project, records)
        report["fixture_kind"] = "real reference manifest, at most 16 pairs" if reference_manifest else "diagnostic pixels/CIF"
        report["checks"].append("Managed raster/CIF import, spatial viewport and inherited head coordinates")
        plan = service.create_work_plan(
            project_id=project.id, title="Deployment probe", principal=session.principal, idempotency_key="plan"
        )
        if service.get_work_plan(project.id, plan.id) != plan:
            raise RuntimeError("Local event-store readback failed")
        report["checks"].append("Local account/project/event-store readback")
        shell = ProjectManagerShell()
        shell.resize(1280, 900)
        shell.set_session_summary("Deployment self-test · local")
        dialogs.append(shell)
        controller = DesktopController(shell, service, session, thumbnail_store_uri="memory://", plugin_items=[])
        controller._agent_timer.stop()
        controller.open_project(ProjectListItem(str(project.id), project.name, project.width, project.height, "local"))
        shell.show()
        verify_workspace(app, controller, report["managed_matrix"], run / "workspace.png")
        advance_head(service, session, report["managed_matrix"], records[0]["vector"], "deployment-visible-head")
        controller._refresh_matrix(controller._workspace, project.id)
        verify_workspace(app, controller, report["managed_matrix"], run / "workspace-updated.png")
        report["checks"].append("Qt managed matrix, raster thumbnail pixels and visible head refresh")
        service = EmbeddedProjectService(run / "profile", secret_store=store)
        verify_project(service, report["managed_matrix"])
        controller.service = service
        controller.open_project(ProjectListItem(str(project.id), project.name, project.width, project.height, "local"))
        verify_workspace(app, controller, report["managed_matrix"], run / "workspace-reopened.png")
        report["checks"].append("Fresh service/project reopen preserves managed versions and Qt thumbnails")
        from kraken_hub.presentation.probes.deployment_legacy_work_probe import probe_legacy_work_orders

        report["legacy_work_order_migration"] = probe_legacy_work_orders(service, session, project, run, app)
        verify_project(service, report["managed_matrix"])
        report["checks"].append("Legacy work Qt preview/import, cached observations, cache replay and verified backup")
        from kraken_hub.presentation.probes.deployment_administration_probe import probe_administration

        report["administration_operations"] = probe_administration(service, session, project, run, app, report["managed_matrix"])
        report["checks"].append("Qt administration backup/scan/restore, GUI heartbeat and busy close guard")
        from types import SimpleNamespace
        from kraken_manager.domain.multilayer import ControlPoint
        from kraken_manager.infrastructure.registration import fit_affine

        coordinates = SimpleNamespace(to_physical=lambda xy: (xy[0] * 1e-6, xy[1] * 1e-6), uncertainty_m=0)
        fit = fit_affine(
            coordinates,
            coordinates,
            tuple(ControlPoint(xy, (xy[0] + 1, xy[1] + 2), 0) for xy in ((0, 0), (1, 0), (0, 1), (1, 1))),
            max_error_m=1e-12,
        )
        if fit.status != "fitted" or fit.max_residual_m > 1e-12:
            raise RuntimeError("Native affine solver deployment probe failed")
        report["checks"].append("NumPy affine solver on declared synthetic control points")
        from hashlib import sha256
        from io import BytesIO
        from PIL import Image
        from kraken_manager.domain.multilayer import ArtifactAnchor, CoordinateCalibration
        from kraken_manager.infrastructure.conductor_masks import extract_conductors

        mask_image = Image.new("L", (4, 4), 0)
        mask_image.putpixel((1, 1), 255)
        mask_image.putpixel((2, 2), 255)
        mask_image.putpixel((3, 0), 128)  # Explicit unobserved pixel, never material.
        buffer = BytesIO()
        mask_image.save(buffer, format="PNG")
        mask_bytes = buffer.getvalue()
        mask_anchor = ArtifactAnchor(str(uuid4()), sha256(mask_bytes).hexdigest())
        mask_calibration = CoordinateCalibration(str(uuid4()), 0, mask_anchor, (1e-6, 1e-6))
        extracted = [extract_conductors(
            mask_bytes, project_id=str(project.id), mask=mask_anchor, calibration=mask_calibration,
            connectivity=adjacency, foreground_value=255, nodata_value=128,
        ) for adjacency in (4, 8)]
        if ([len(result.components) for result in extracted] != [2, 1]
                or any(result.nodata_runs != ((0, 3, 4),) for result in extracted)
                or sum(component.pixel_count for component in extracted[1].components) != 2):
            raise RuntimeError("SciPy binary component/void deployment probe failed")
        (run / "synthetic-binary-mask.png").write_bytes(mask_bytes)
        report["synthetic_mask_sha256"] = mask_anchor.sha256
        report["checks"].append("SciPy exact 4/8 binary components and explicit nodata handling")
        from kraken_hub.presentation.probes.deployment_stitch_probe import probe_stitch_worker
        report["stitch_workers"] = probe_stitch_worker(run / "stitch-workers")
        report["checks"].append("Private raster/vector stitch worker: pixels, coverage, CIF hole and rotated box")
        layer = service.list_layers(project.id)[0]
        for label, dialog in (
            ("planning", WorkPlanningDialog(service, session.principal, project.id, language="English")),
            ("pipeline", LayerPipelineDialog(service, session.principal, project.id, layer.id, language="English")),
            ("quality", QualityDialog(service, session.principal, project.id)),
        ):
            dialogs.append(dialog)
            dialog.show()
            deadline = time.monotonic() + 15
            while getattr(dialog, "_worker", None) is not None and time.monotonic() < deadline:
                app.processEvents()
                time.sleep(0.01)
            if getattr(dialog, "_worker", None) is not None:
                raise RuntimeError(f"{label} initialization timed out")
            app.processEvents()
            if not dialog.grab().save(str(run / f"{label}.png")):
                raise RuntimeError(f"{label} rendering failed")
            report["checks"].append(f"{label} service load and Qt rendering")
            dialog.close()
        controller.agent_runtime.ensure_started()
        if controller.agent_runtime._request("GET", "/api/v1/health")["status"] != "ok":
            raise RuntimeError("Local Agent health check failed")
        controller.agent_runtime.shutdown()
        report["checks"].append("Bundled/local Agent launch, authenticated health and graceful shutdown")
        for record in records:
            for role in ("image", "vector"):
                if sha256(Path(record[role]).read_bytes()).hexdigest() != record[role + "_sha256"]:
                    raise RuntimeError("Deployment probe source changed during the run")
        report["source_files_unchanged"] = True
        if reference_manifest:
            report["reference_manifest_sha256"] = sha256(Path(reference_manifest).read_bytes()).hexdigest()
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=str(exc), traceback=traceback.format_exc())
    finally:
        if controller is not None:
            controller._agent_timer.stop()
            try:
                controller.agent_runtime.shutdown()
            except Exception as exc:
                report.update(status="failed", cleanup_error=str(exc))
        for dialog in reversed(dialogs):
            worker = getattr(dialog, "_worker", None)
            if worker is not None and worker.isRunning():
                worker.wait(15000)
            dialog.close()
        if app is not None:
            app.processEvents()
        if store is not None:
            for key in store.created:
                try:
                    store.delete(key)
                except Exception as exc:
                    report.update(status="failed", secret_cleanup_error=str(exc))
        (run / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if report["status"] == "passed" else 1
